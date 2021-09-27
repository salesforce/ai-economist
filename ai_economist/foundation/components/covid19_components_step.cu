// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

extern "C" {
    // CUDA version of the components in
    // "ai_economist.foundation.components.covid19_components.py"
    __global__ void CudaControlUSStateOpenCloseStatusStep(
        int * stringency_level,
        const int kActionCooldownPeriod,
        int * action_in_cooldown_until,
        const int * kDefaultAgentActionMask,
        const int * kNoOpAgentActionMask,
        const int kNumStringencyLevels,
        int * actions,
        float * obs_a_stringency_policy_indicators,
        float * obs_a_action_mask,
        float * obs_p_stringency_policy_indicators,
        int * env_timestep_arr,
        const int kNumAgents,
        const int kEpisodeLength
    ) {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        // Increment time ONCE -- only 1 thread can do this.
        if (kAgentId == 0) {
            env_timestep_arr[kEnvId] += 1;
        }

        // Wait here until timestep has been updated
        __syncthreads();

        assert(env_timestep_arr[kEnvId] > 0 &&
            env_timestep_arr[kEnvId] <= kEpisodeLength);
        assert (kAgentId <= kNumAgents - 1);

        // Update the stringency levels for the US states
        if (kAgentId < (kNumAgents - 1)) {
            // Indices for time-dependent and time-independent arrays
            // Time dependent arrays have shapes
            // (num_envs, kEpisodeLength + 1, kNumAgents - 1)
            // Time independent arrays have shapes (num_envs, kNumAgents - 1)
            const int kArrayIdxOffset = kEnvId * (kEpisodeLength + 1) *
                (kNumAgents - 1);
            int time_dependent_array_index_curr_t = kArrayIdxOffset +
                env_timestep_arr[kEnvId] * (kNumAgents - 1) + kAgentId;
            int time_dependent_array_index_prev_t = kArrayIdxOffset +
                (env_timestep_arr[kEnvId] - 1) * (kNumAgents - 1) + kAgentId;
            const int time_independent_array_index = kEnvId * (kNumAgents - 1) +
                kAgentId;

            // action is not a NO-OP
            if (actions[time_independent_array_index] != 0) {
                stringency_level[time_dependent_array_index_curr_t] =
                    actions[time_independent_array_index];
            } else {
                stringency_level[time_dependent_array_index_curr_t] =
                    stringency_level[time_dependent_array_index_prev_t];
            }

            if (env_timestep_arr[kEnvId] == action_in_cooldown_until[
                time_independent_array_index] + 1) {
                if (actions[time_independent_array_index] != 0) {
                    assert(0 <= actions[time_independent_array_index] <=
                        kNumStringencyLevels);
                    action_in_cooldown_until[time_independent_array_index] +=
                        kActionCooldownPeriod;
                } else {
                    action_in_cooldown_until[time_independent_array_index] += 1;
                }
            }

            obs_a_stringency_policy_indicators[
                time_independent_array_index
            ] = stringency_level[time_dependent_array_index_curr_t] /
                static_cast<float>(kNumStringencyLevels);

            // CUDA version of generate_masks()
            for (int action_id = 0; action_id < (kNumStringencyLevels + 1);
                action_id++) {
                int action_mask_array_index =
                    kEnvId * (kNumStringencyLevels + 1) *
                    (kNumAgents - 1) + action_id * (kNumAgents - 1) + kAgentId;
                if (env_timestep_arr[kEnvId] < action_in_cooldown_until[
                    time_independent_array_index]
                ) {
                    obs_a_action_mask[action_mask_array_index] =
                    kNoOpAgentActionMask[action_id];
                } else {
                    obs_a_action_mask[action_mask_array_index] =
                    kDefaultAgentActionMask[action_id];
                }
            }
        }

        // Update planner obs after all the agents' obs are updated
        __syncthreads();

        if (kAgentId == kNumAgents - 1) {
            for (int ag_id = 0; ag_id < (kNumAgents - 1); ag_id++) {
                const int kIndex = kEnvId * (kNumAgents - 1) + ag_id;
                obs_p_stringency_policy_indicators[
                    kIndex
                ] = 
                    obs_a_stringency_policy_indicators[
                        kIndex
                    ];
            }
        }
    }

    __global__ void CudaFederalGovernmentSubsidyStep(
        int * subsidy_level,
        float * subsidy,
        const int kSubsidyInterval,
        const int kNumSubsidyLevels,
        const float * KMaxDailySubsidyPerState,
        const int * kDefaultPlannerActionMask,
        const int * kNoOpPlannerActionMask,
        int * actions,
        float * obs_a_time_until_next_subsidy,
        float * obs_a_current_subsidy_level,
        float * obs_p_time_until_next_subsidy,
        float * obs_p_current_subsidy_level,
        float * obs_p_action_mask,
        int * env_timestep_arr,
        const int kNumAgents,
        const int kEpisodeLength
    ) {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        assert(env_timestep_arr[kEnvId] > 0 &&
            env_timestep_arr[kEnvId] <= kEpisodeLength);
        assert (kAgentId <= kNumAgents - 1);

        int t_since_last_subsidy = env_timestep_arr[kEnvId] %
            kSubsidyInterval;

        // Setting the (federal government) planner's subsidy level
        // to be the subsidy level for all the US states
        if (kAgentId < kNumAgents - 1) {
            // Indices for time-dependent and time-independent arrays
            // Time dependent arrays have shapes (num_envs,
            // kEpisodeLength + 1, kNumAgents - 1)
            // Time independent arrays have shapes (num_envs, kNumAgents - 1)
            const int kArrayIdxOffset = kEnvId * (kEpisodeLength + 1) *
                (kNumAgents - 1);
            int time_dependent_array_index_curr_t = kArrayIdxOffset +
                env_timestep_arr[kEnvId] * (kNumAgents - 1) + kAgentId;
            int time_dependent_array_index_prev_t = kArrayIdxOffset +
                (env_timestep_arr[kEnvId] - 1) * (kNumAgents - 1) + kAgentId;
            const int time_independent_array_index = kEnvId *
                (kNumAgents - 1) + kAgentId;

            if ((env_timestep_arr[kEnvId] - 1) % kSubsidyInterval == 0) {
                assert(0 <= actions[kEnvId] <= kNumSubsidyLevels);
                subsidy_level[time_dependent_array_index_curr_t] =
                    actions[kEnvId];
            } else {
                subsidy_level[time_dependent_array_index_curr_t] =
                    subsidy_level[time_dependent_array_index_prev_t];
            }
            // Setting the subsidies for the US states
            // based on the federal government's subsidy level
            subsidy[time_dependent_array_index_curr_t] =
                subsidy_level[time_dependent_array_index_curr_t] *
                KMaxDailySubsidyPerState[kAgentId] / kNumSubsidyLevels;

            obs_a_time_until_next_subsidy[
                time_independent_array_index] =
                    1 - (t_since_last_subsidy /
                    static_cast<float>(kSubsidyInterval));
            obs_a_current_subsidy_level[
                time_independent_array_index] =
                    subsidy_level[time_dependent_array_index_curr_t] /
                    static_cast<float>(kNumSubsidyLevels);
        } else if (kAgentId == (kNumAgents - 1)) {
            for (int action_id = 0; action_id < kNumSubsidyLevels + 1;
                action_id++) {
                int action_mask_array_index = kEnvId *
                    (kNumSubsidyLevels + 1) + action_id;
                if (env_timestep_arr[kEnvId] % kSubsidyInterval == 0) {
                    obs_p_action_mask[action_mask_array_index] =
                        kDefaultPlannerActionMask[action_id];
                } else {
                    obs_p_action_mask[action_mask_array_index] =
                        kNoOpPlannerActionMask[action_id];
                }
            }
            // Update planner obs after the agent's obs are updated
            __syncthreads();

            if (kAgentId == (kNumAgents - 1)) {
                // Just use the values for agent id 0
                obs_p_time_until_next_subsidy[kEnvId] =
                    obs_a_time_until_next_subsidy[
                        kEnvId * (kNumAgents - 1)
                    ];
                obs_p_current_subsidy_level[kEnvId] = 
                    obs_a_current_subsidy_level[
                        kEnvId * (kNumAgents - 1)
                    ];
            }
        }
    }

    __global__ void CudaVaccinationCampaignStep(
        int * vaccinated,
        const int * kNumVaccinesPerDelivery,
        int * num_vaccines_available_t,
        const int kDeliveryInterval,
        const int kTimeWhenVaccineDeliveryBegins,
        float * obs_a_vaccination_campaign_t_until_next_vaccines,
        float * obs_p_vaccination_campaign_t_until_next_vaccines,
        int * env_timestep_arr,
        int kNumAgents,
        int kEpisodeLength
    ) {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <=
            kEpisodeLength);
        assert(kTimeWhenVaccineDeliveryBegins > 0);
        assert (kAgentId <= kNumAgents - 1);

        // CUDA version of generate observations()
        int t_first_delivery = kTimeWhenVaccineDeliveryBegins +
            kTimeWhenVaccineDeliveryBegins % kDeliveryInterval;
        int next_t = env_timestep_arr[kEnvId] + 1;
        float t_until_next_vac;
        if (next_t <= t_first_delivery) {
            t_until_next_vac = min(
                1,
                (t_first_delivery - next_t) / kDeliveryInterval);
        } else {
            float t_since_last_vac = next_t % kDeliveryInterval;
            t_until_next_vac = 1 - (t_since_last_vac / kDeliveryInterval);
        }

        // Update the vaccinated numbers for just the US states
        if (kAgentId < (kNumAgents - 1)) {
            const int time_independent_array_index = kEnvId *
                (kNumAgents - 1) + kAgentId;
            if ((env_timestep_arr[kEnvId] >= kTimeWhenVaccineDeliveryBegins) &&
                (env_timestep_arr[kEnvId] % kDeliveryInterval == 0)) {
                num_vaccines_available_t[time_independent_array_index] =
                    kNumVaccinesPerDelivery[kAgentId];
            } else {
                num_vaccines_available_t[time_independent_array_index] = 0;
            }
            obs_a_vaccination_campaign_t_until_next_vaccines[
                time_independent_array_index] = t_until_next_vac;
        } else if (kAgentId == kNumAgents - 1) {
            obs_p_vaccination_campaign_t_until_next_vaccines[kEnvId] =
            t_until_next_vac;
        }
    }
}
