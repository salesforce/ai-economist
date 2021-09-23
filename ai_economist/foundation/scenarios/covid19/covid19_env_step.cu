// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

__constant__ float kEpsilon = 1.0e-10;  // used to prevent division by 0

extern "C" {
// CUDA version of the scenario_step() in
// "ai_economist.foundation.scenarios.covid19_env.py"

    // CUDA version of the sir_step() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ void cuda_sir_step(
        float* susceptible,
        float* infected,
        float* recovered,
        float* vaccinated,
        float* deaths,
        int* num_vaccines_available_t,
        const int* kRealWorldStringencyPolicyHistory,
        const float kStatePopulation,
        const int kNumAgents,
        const int kBetaDelay,
        const float kBetaSlope,
        const float kbetaIntercept,
        int* stringency_level,
        float* beta,
        const float kGamma,
        const float kDeathRate,
        const int kEnvId,
        const int kAgentId,
        int timestep,
        const int kEpisodeLength,
        const int kArrayIdxCurrentTime,
        const int kArrayIdxPrevTime,
        const int kTimeIndependentArrayIdx
    ) {
        float susceptible_fraction_vaccinated = min(
            1.0,
            num_vaccines_available_t[kTimeIndependentArrayIdx] /
                (susceptible[kArrayIdxPrevTime] + kEpsilon));
        float vaccinated_t = min(
            static_cast<float>(num_vaccines_available_t[
                kTimeIndependentArrayIdx]),
            susceptible[kArrayIdxPrevTime]);

        // (S/N) * I in place of (S*I) / N to prevent overflow
        float neighborhood_SI_over_N = susceptible[kArrayIdxPrevTime] /
            kStatePopulation * infected[kArrayIdxPrevTime];
        int stringency_level_tmk;
        if (timestep < kBetaDelay) {
            stringency_level_tmk = kRealWorldStringencyPolicyHistory[
                (timestep - 1) * (kNumAgents - 1) + kAgentId];
        } else {
            stringency_level_tmk = stringency_level[kEnvId * (
                kEpisodeLength + 1) * (kNumAgents - 1) +
                (timestep - kBetaDelay) * (kNumAgents - 1) + kAgentId];
        }
        beta[kTimeIndependentArrayIdx] = stringency_level_tmk *
            kBetaSlope + kbetaIntercept;

        float dS_t = -(neighborhood_SI_over_N * beta[
            kTimeIndependentArrayIdx] *
            (1 - susceptible_fraction_vaccinated) + vaccinated_t);
        float dR_t = kGamma * infected[kArrayIdxPrevTime] + vaccinated_t;
        float dI_t = - dS_t - dR_t;

        susceptible[kArrayIdxCurrentTime] = max(
            0.0,
            susceptible[kArrayIdxPrevTime] + dS_t);
        infected[kArrayIdxCurrentTime] = max(
            0.0,
            infected[kArrayIdxPrevTime] + dI_t);
        recovered[kArrayIdxCurrentTime] = max(
            0.0,
            recovered[kArrayIdxPrevTime] + dR_t);

        vaccinated[kArrayIdxCurrentTime] = vaccinated_t +
            vaccinated[kArrayIdxPrevTime];
        float recovered_but_not_vaccinated = recovered[kArrayIdxCurrentTime] -
            vaccinated[kArrayIdxCurrentTime];
        deaths[kArrayIdxCurrentTime] = recovered_but_not_vaccinated *
            kDeathRate;
    }

    // CUDA version of the softplus() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ float softplus(float x) {
        const float kBeta = 1.0;
        const float kThreshold = 20.0;
        if (kBeta * x < kThreshold) {
            return 1.0 / kBeta * log(1.0 + exp(kBeta * x));
        } else {
            return x;
        }
    }

    __device__ float signal2unemployment(
        const int kEnvId,
        const int kAgentId,
        float* signal,
        const float* kUnemploymentConvolutionalFilters,
        const float kUnemploymentBias,
        const int kNumAgents,
        const int kFilterLen,
        const int kNumFilters
    ) {
        float unemployment = 0.0;
        const int kArrayIndexOffset = kEnvId * (kNumAgents - 1) * kNumFilters *
            kFilterLen + kAgentId * kNumFilters * kFilterLen;
        for (int index = 0; index < (kFilterLen * kNumFilters); index ++) {
            unemployment += signal[kArrayIndexOffset + index] *
            kUnemploymentConvolutionalFilters[index];
        }
        return softplus(unemployment) + kUnemploymentBias;
    }

    // CUDA version of the unemployment_step() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ void cuda_unemployment_step(
        float* unemployed,
        int* stringency_level,
        int* delta_stringency_level,
        const float* kGroupedConvolutionalFilterWeights,
        const float* kUnemploymentConvolutionalFilters,
        const float* kUnemploymentBias,
        float* convolved_signal,
        const int kFilterLen,
        const int kNumFilters,
        const float kStatePopulation,
        const int kNumAgents,
        const int kEnvId,
        const int kAgentId,
        int timestep,
        const int kArrayIdxCurrentTime,
        const int kArrayIdxPrevTime
    ) {
        // Shift array by kNumAgents - 1
        for (int idx = 0; idx < kFilterLen - 1; idx ++) {
            delta_stringency_level[
                kEnvId * kFilterLen * (kNumAgents - 1) + idx *
                (kNumAgents - 1) + kAgentId
            ] =
            delta_stringency_level[
                kEnvId * kFilterLen * (kNumAgents - 1) + (idx + 1) *
                (kNumAgents - 1) + kAgentId
            ];
        }

        delta_stringency_level[
            kEnvId * kFilterLen * (kNumAgents - 1) + (kFilterLen - 1) *
            (kNumAgents - 1) + kAgentId
        ] = stringency_level[kArrayIdxCurrentTime] -
            stringency_level[kArrayIdxPrevTime];

        // convolved_signal refers to the convolution between the filter weights
        // and the delta stringency levels
        for (int filter_idx = 0; filter_idx < kNumFilters; filter_idx ++) {
            for (int idx = 0; idx < kFilterLen; idx ++) {
                convolved_signal[
                    kEnvId * (kNumAgents - 1) * kNumFilters * kFilterLen +
                    kAgentId * kNumFilters * kFilterLen +
                    filter_idx * kFilterLen +
                    idx
                ] =
                delta_stringency_level[kEnvId * kFilterLen * (kNumAgents - 1) +
                    idx * (kNumAgents - 1) + kAgentId] *
                kGroupedConvolutionalFilterWeights[kAgentId * kNumFilters +
                    filter_idx];
            }
        }

        float unemployment_rate = signal2unemployment(
            kEnvId,
            kAgentId,
            convolved_signal,
            kUnemploymentConvolutionalFilters,
            kUnemploymentBias[kAgentId],
            kNumAgents,
            kFilterLen,
            kNumFilters);

        unemployed[kArrayIdxCurrentTime] =
            unemployment_rate * kStatePopulation / 100.0;
    }

    // CUDA version of the economy_step() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ void cuda_economy_step(
        float* infected,
        float* deaths,
        float* unemployed,
        float* incapacitated,
        float* cant_work,
        float* num_people_that_can_work,
        const float kStatePopulation,
        const float kInfectionTooSickToWorkRate,
        const float kPopulationBetweenAge18And65,
        const float kDailyProductionPerWorker,
        float* productivity,
        float* subsidy,
        float* postsubsidy_productivity,
        int timestep,
        const int kArrayIdxCurrentTime,
        int kTimeIndependentArrayIdx
    ) {
        incapacitated[kTimeIndependentArrayIdx] =
            kInfectionTooSickToWorkRate * infected[kArrayIdxCurrentTime] +
            deaths[kArrayIdxCurrentTime];
        cant_work[kTimeIndependentArrayIdx] =
            incapacitated[kTimeIndependentArrayIdx] *
            kPopulationBetweenAge18And65 + unemployed[kArrayIdxCurrentTime];
        int num_workers = static_cast<int>(kStatePopulation) * kPopulationBetweenAge18And65;
        num_people_that_can_work[kTimeIndependentArrayIdx] = max(
            0.0,
            num_workers - cant_work[kTimeIndependentArrayIdx]);
        productivity[kArrayIdxCurrentTime] =
            num_people_that_can_work[kTimeIndependentArrayIdx] *
            kDailyProductionPerWorker;

        postsubsidy_productivity[kArrayIdxCurrentTime] =
            productivity[kArrayIdxCurrentTime] +
            subsidy[kArrayIdxCurrentTime];
    }

    // CUDA version of crra_nonlinearity() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ float crra_nonlinearity(
        float x,
        const float kEta,
        const int kNumDaysInAnYear
    ) {
        float annual_x = kNumDaysInAnYear * x;
        float annual_x_clipped = annual_x;
        if (annual_x < 0.1) {
            annual_x_clipped = 0.1;
        } else if (annual_x > 3.0) {
            annual_x_clipped = 3.0;
        }
        float annual_crra = 1 + (pow(annual_x_clipped, (1 - kEta)) - 1) /
            (1 - kEta);
        float daily_crra = annual_crra / kNumDaysInAnYear;
        return daily_crra;
    }

    // CUDA version of min_max_normalization() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ float min_max_normalization(
        float x,
        const float kMinX,
        const float kMaxX
    ) {
        return (x - kMinX) / (kMaxX - kMinX + kEpsilon);
    }

    // CUDA version of get_rew() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __device__ float get_rew(
        const float kHealthIndexWeightage,
        float health_index,
        const float kEconomicIndexWeightage,
        float economic_index
    ) {
        return (
            kHealthIndexWeightage * health_index
            + kEconomicIndexWeightage * economic_index) /
            (kHealthIndexWeightage + kEconomicIndexWeightage);
    }

    // CUDA version of scenario_step() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __global__ void CudaCovidAndEconomySimulationStep(
        float* susceptible,
        float* infected,
        float* recovered,
        float* deaths,
        float* vaccinated,
        float* unemployed,
        float* subsidy,
        float* productivity,
        int* stringency_level,
        const int kNumStringencyLevels,
        float* postsubsidy_productivity,
        int* num_vaccines_available_t,
        const int* kRealWorldStringencyPolicyHistory,
        const int kBetaDelay,
        const float* kBetaSlopes,
        const float* kbetaIntercepts,
        float* beta,
        const float kGamma,
        const float kDeathRate,
        float* incapacitated,
        float* cant_work,
        float* num_people_that_can_work,
        const int* us_kStatePopulation,
        const float kInfectionTooSickToWorkRate,
        const float kPopulationBetweenAge18And65,
        const int kFilterLen,
        const int kNumFilters,
        int* delta_stringency_level,
        const float* kGroupedConvolutionalFilterWeights,
        const float* kUnemploymentConvolutionalFilters,
        const float* kUnemploymentBias,
        float* signal,
        const float kDailyProductionPerWorker,
        const float* maximum_productivity,
        float* obs_a_world_agent_state,
        float* obs_a_world_agent_postsubsidy_productivity,
        float* obs_a_world_lagged_stringency_level,
        float* obs_a_time,
        float* obs_p_world_agent_state,
        float* obs_p_world_agent_postsubsidy_productivity,
        float* obs_p_world_lagged_stringency_level,
        float* obs_p_time,
        int * env_timestep_arr,
        const int kNumAgents,
        const int kEpisodeLength
    ) {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        assert(env_timestep_arr[kEnvId] > 0 &&
            env_timestep_arr[kEnvId] <= kEpisodeLength);
        assert (kAgentId <= kNumAgents - 1);
        const int kNumFeatures = 6;

        if (kAgentId < (kNumAgents - 1)) {
            // Indices for time-dependent and time-independent arrays
            // Time dependent arrays have shapes (num_envs,
            // kEpisodeLength + 1, kNumAgents - 1)
            // Time independent arrays have shapes (num_envs, kNumAgents - 1)
            const int kArrayIndexOffset = kEnvId * (kEpisodeLength + 1) *
                (kNumAgents - 1);
            int kArrayIdxCurrentTime = kArrayIndexOffset +
                env_timestep_arr[kEnvId] * (kNumAgents - 1) + kAgentId;
            int kArrayIdxPrevTime = kArrayIndexOffset +
                (env_timestep_arr[kEnvId] - 1) * (kNumAgents - 1) + kAgentId;
            const int kTimeIndependentArrayIdx = kEnvId *
                (kNumAgents - 1) + kAgentId;

            const float kStatePopulation = static_cast<float>(us_kStatePopulation[kAgentId]);

            cuda_sir_step(
                susceptible,
                infected,
                recovered,
                vaccinated,
                deaths,
                num_vaccines_available_t,
                kRealWorldStringencyPolicyHistory,
                kStatePopulation,
                kNumAgents,
                kBetaDelay,
                kBetaSlopes[kAgentId],
                kbetaIntercepts[kAgentId],
                stringency_level,
                beta,
                kGamma,
                kDeathRate,
                kEnvId,
                kAgentId,
                env_timestep_arr[kEnvId],
                kEpisodeLength,
                kArrayIdxCurrentTime,
                kArrayIdxPrevTime,
                kTimeIndependentArrayIdx);

            cuda_unemployment_step(
                unemployed,
                stringency_level,
                delta_stringency_level,
                kGroupedConvolutionalFilterWeights,
                kUnemploymentConvolutionalFilters,
                kUnemploymentBias,
                signal,
                kFilterLen,
                kNumFilters,
                kStatePopulation,
                kNumAgents,
                kEnvId,
                kAgentId,
                env_timestep_arr[kEnvId],
                kArrayIdxCurrentTime,
                kArrayIdxPrevTime);

            cuda_economy_step(
                infected,
                deaths,
                unemployed,
                incapacitated,
                cant_work,
                num_people_that_can_work,
                kStatePopulation,
                kInfectionTooSickToWorkRate,
                kPopulationBetweenAge18And65,
                kDailyProductionPerWorker,
                productivity,
                subsidy,
                postsubsidy_productivity,
                env_timestep_arr[kEnvId],
                kArrayIdxCurrentTime,
                kTimeIndependentArrayIdx);

            // CUDA version of generate observations
            // Agents' observations
            int kFeatureArrayIndexOffset = kEnvId * kNumFeatures *
                (kNumAgents - 1) + kAgentId;
            obs_a_world_agent_state[
                kFeatureArrayIndexOffset + 0 * (kNumAgents - 1)
            ] = susceptible[kArrayIdxCurrentTime] / kStatePopulation;
            obs_a_world_agent_state[
                kFeatureArrayIndexOffset + 1 * (kNumAgents - 1)
            ] = infected[kArrayIdxCurrentTime] / kStatePopulation;
            obs_a_world_agent_state[
                kFeatureArrayIndexOffset + 2 * (kNumAgents - 1)
            ] = recovered[kArrayIdxCurrentTime] / kStatePopulation;
            obs_a_world_agent_state[
                kFeatureArrayIndexOffset + 3 * (kNumAgents - 1)
            ] = deaths[kArrayIdxCurrentTime] / kStatePopulation;
            obs_a_world_agent_state[
                kFeatureArrayIndexOffset + 4 * (kNumAgents - 1)
            ] = vaccinated[kArrayIdxCurrentTime] / kStatePopulation;
            obs_a_world_agent_state[
                kFeatureArrayIndexOffset + 5 * (kNumAgents - 1)
            ] = unemployed[kArrayIdxCurrentTime] / kStatePopulation;

            for (int feature_id = 0; feature_id < kNumFeatures; feature_id ++) {
                const int kIndex = feature_id * (kNumAgents - 1);
                obs_p_world_agent_state[kFeatureArrayIndexOffset +
                    kIndex
                ] = obs_a_world_agent_state[kFeatureArrayIndexOffset +
                    kIndex];
            }

            obs_a_world_agent_postsubsidy_productivity[
                kTimeIndependentArrayIdx
            ] = postsubsidy_productivity[kArrayIdxCurrentTime] /
                maximum_productivity[kAgentId];
            obs_p_world_agent_postsubsidy_productivity[
                kTimeIndependentArrayIdx
            ] = obs_a_world_agent_postsubsidy_productivity[
                    kTimeIndependentArrayIdx
                ];

            int t_beta = env_timestep_arr[kEnvId] - kBetaDelay + 1;
            if (t_beta < 0) {
                obs_a_world_lagged_stringency_level[
                    kTimeIndependentArrayIdx
                ] = kRealWorldStringencyPolicyHistory[
                        env_timestep_arr[kEnvId] * (kNumAgents - 1) + kAgentId
                    ] / static_cast<float>(kNumStringencyLevels);
            } else {
                obs_a_world_lagged_stringency_level[
                    kTimeIndependentArrayIdx
                ] = stringency_level[
                        kArrayIndexOffset +
                        t_beta * (kNumAgents - 1) +
                        kAgentId
                    ] / static_cast<float>(kNumStringencyLevels);
            }
            obs_p_world_lagged_stringency_level[
                kTimeIndependentArrayIdx
            ] = obs_a_world_lagged_stringency_level[
                    kTimeIndependentArrayIdx];
            // Below, we assume observation scaling = True
            // (otherwise, 'obs_a_time[kTimeIndependentArrayIdx] =
            // static_cast<float>(env_timestep_arr[kEnvId])
            obs_a_time[kTimeIndependentArrayIdx] =
                env_timestep_arr[kEnvId] / static_cast<float>(kEpisodeLength);
        } else if (kAgentId == kNumAgents - 1) {
            obs_p_time[kEnvId] = env_timestep_arr[kEnvId] /
            static_cast<float>(kEpisodeLength);
        }
    }

    // CUDA version of the compute_reward() in
    // "ai_economist.foundation.scenarios.covid19_env.py"
    __global__ void CudaComputeReward(
        float* rewards_a,
        float* rewards_p,
        const int kNumDaysInAnYear,
        const int kValueOfLife,
        const float kRiskFreeInterestRate,
        const float kEconomicRewardCrraEta,
        const float* kMinMarginalAgentHealthIndex,
        const float* kMaxMarginalAgentHealthIndex,
        const float* kMinMarginalAgentEconomicIndex,
        const float* kMaxMarginalAgentEconomicIndex,
        const float kMinMarginalPlannerHealthIndex,
        const float kMaxMarginalPlannerHealthIndex,
        const float kMinMarginalPlannerEconomicIndex,
        const float kMaxMarginalPlannerEconomicIndex,
        const float* kWeightageOnMarginalAgentHealthIndex,
        const float* kWeightageOnMarginalPlannerHealthIndex,
        const float kWeightageOnMarginalAgentEconomicIndex,
        const float kWeightageOnMarginalPlannerEconomicIndex,
        const float* kAgentsHealthNorm,
        const float* kAgentsEconomicNorm,
        const float kPlannerHealthNorm,
        const float kPlannerEconomicNorm,
        float* deaths,
        float* subsidy,
        float* postsubsidy_productivity,
        int* env_done_arr,
        int* env_timestep_arr,
        const int kNumAgents,
        const int kEpisodeLength
    ) {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        assert(env_timestep_arr[kEnvId] > 0 &&
             env_timestep_arr[kEnvId] <= kEpisodeLength);
        assert (kAgentId <= kNumAgents - 1);

        const int kArrayIndexOffset = kEnvId * (kEpisodeLength + 1) *
            (kNumAgents - 1);
        if (kAgentId < (kNumAgents - 1)) {
            // Agents' rewards
            // Indices for time-dependent and time-independent arrays
            // Time dependent arrays have shapes (num_envs,
            // kEpisodeLength + 1, kNumAgents - 1)
            // Time independent arrays have shapes (num_envs, kNumAgents - 1)
            int kArrayIdxCurrentTime = kArrayIndexOffset +
                env_timestep_arr[kEnvId] * (kNumAgents - 1) + kAgentId;
            int kArrayIdxPrevTime = kArrayIndexOffset +
                (env_timestep_arr[kEnvId] - 1) * (kNumAgents - 1) + kAgentId;
            const int kTimeIndependentArrayIdx = kEnvId *
                (kNumAgents - 1) + kAgentId;

            float marginal_deaths = deaths[kArrayIdxCurrentTime] -
                deaths[kArrayIdxPrevTime];

            // Note: changing the order of operations to prevent overflow
            float marginal_agent_health_index = - marginal_deaths /
                (kAgentsHealthNorm[kAgentId] /
                static_cast<float>(kValueOfLife));

            float marginal_agent_economic_index = crra_nonlinearity(
                postsubsidy_productivity[kArrayIdxCurrentTime] /
                kAgentsEconomicNorm[kAgentId],
                kEconomicRewardCrraEta,
                kNumDaysInAnYear);

            marginal_agent_health_index = min_max_normalization(
                marginal_agent_health_index,
                kMinMarginalAgentHealthIndex[kAgentId],
                kMaxMarginalAgentHealthIndex[kAgentId]);
            marginal_agent_economic_index = min_max_normalization(
                marginal_agent_economic_index,
                kMinMarginalAgentEconomicIndex[kAgentId],
                kMaxMarginalAgentEconomicIndex[kAgentId]);

            rewards_a[kTimeIndependentArrayIdx] = get_rew(
                kWeightageOnMarginalAgentHealthIndex[kAgentId],
                marginal_agent_health_index,
                kWeightageOnMarginalPlannerHealthIndex[kAgentId],
                marginal_agent_economic_index);
        } else if (kAgentId == kNumAgents - 1) {
            // Planner's rewards
            float total_marginal_deaths = 0;
            for (int ag_id = 0; ag_id < (kNumAgents - 1); ag_id ++) {
                total_marginal_deaths += (
                    deaths[kArrayIndexOffset + env_timestep_arr[kEnvId] *
                        (kNumAgents - 1) + ag_id] -
                    deaths[kArrayIndexOffset + (env_timestep_arr[kEnvId] - 1) *
                        (kNumAgents - 1) + ag_id]);
            }
            // Note: changing the order of operations to prevent overflow
            float marginal_planner_health_index = -total_marginal_deaths /
                (kPlannerHealthNorm / static_cast<float>(kValueOfLife));

            float total_subsidy = 0.0;
            float total_postsubsidy_productivity = 0.0;
            for (int ag_id = 0; ag_id < (kNumAgents - 1); ag_id ++) {
                total_subsidy += subsidy[kArrayIndexOffset +
                    env_timestep_arr[kEnvId] * (kNumAgents - 1) + ag_id];
                total_postsubsidy_productivity +=
                    postsubsidy_productivity[kArrayIndexOffset +
                    env_timestep_arr[kEnvId] * (kNumAgents - 1) + ag_id];
            }

            float cost_of_subsidy = (1 + kRiskFreeInterestRate) *
                total_subsidy;
            float marginal_planner_economic_index = crra_nonlinearity(
                (total_postsubsidy_productivity - cost_of_subsidy) /
                    kPlannerEconomicNorm,
                kEconomicRewardCrraEta,
                kNumDaysInAnYear);

            marginal_planner_health_index = min_max_normalization(
                marginal_planner_health_index,
                kMinMarginalPlannerHealthIndex,
                kMaxMarginalPlannerHealthIndex);
            marginal_planner_economic_index = min_max_normalization(
                marginal_planner_economic_index,
                kMinMarginalPlannerEconomicIndex,
                kMaxMarginalPlannerEconomicIndex);

            rewards_p[kEnvId] = get_rew(
                kWeightageOnMarginalAgentEconomicIndex,
                marginal_planner_health_index,
                kWeightageOnMarginalPlannerEconomicIndex,
                marginal_planner_economic_index);
        }

        // Wait here for all agents to finish computing rewards
        __syncthreads();

        // Use only agent 0's thread to set done_arr
        if (kAgentId == 0) {
            if (env_timestep_arr[kEnvId] == kEpisodeLength) {
                env_timestep_arr[kEnvId] = 0;
                env_done_arr[kEnvId] = 1;
            }
        }
    }
}
