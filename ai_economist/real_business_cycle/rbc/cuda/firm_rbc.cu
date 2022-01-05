// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause


// Real Business Chain implementation in CUDA C

#include <curand_kernel.h>
#include <math.h>

typedef enum {
  kConsumerType,
  kFirmType,
  kGovernmentType,
} AgentType;

const size_t kBatchSize = M_BATCHSIZE;
const size_t kNumConsumers = M_NUMCONSUMERS;
const bool kCountFirmReward = M_COUNTFIRMREWARD;
const size_t kNumFirms = M_NUMFIRMS;
const size_t kNumGovts = M_NUMGOVERNMENTS;
const float kMaxTime = M_MAXTIME;
const float kCrraParam = M_CRRA_PARAM;
const float kInterestRate = M_INTERESTRATE;
const size_t kNumAgents = kNumConsumers + kNumFirms + kNumGovts;
const bool kIncentivizeFirmActivity = M_SHOULDBOOSTFIRMREWARD;
const float kFirmBoostRewardFactor = M_BOOSTFIRMREWARDFACTOR;
const bool kUseImporter = M_USEIMPORTER;
const float kImporterPrice = M_IMPORTERPRICE;
const float kImporterQuantity = M_IMPORTERQUANTITY;
const float kLaborFloor = M_LABORFLOOR;

// Global state =
const size_t kNumPrices = kNumFirms;  // - prices,
const size_t kNumWages = kNumFirms;  // - wages,
const size_t kNumInventories = kNumFirms;  // - stocks,
const size_t kNumOverdemandFlags = kNumFirms;  // - good overdemanded flag,
const size_t kNumCorporateTaxes = kNumGovts;  // - corporate tax rate
const size_t kNumIncomeTaxes = kNumGovts;  // - income tax rate
const size_t kNumTimeDimensions = 1;  // - time step
const size_t kGlobalStateSize = kNumPrices + kNumWages + kNumInventories + kNumOverdemandFlags + kNumCorporateTaxes + kNumIncomeTaxes + kNumTimeDimensions;

const size_t kIdxPricesOffset = 0;
const size_t kIdxWagesOffset = kNumPrices;
const size_t kIdxStockOffset = kIdxWagesOffset + kNumInventories;
const size_t kIdxOverdemandOffset = kIdxStockOffset + kNumOverdemandFlags;
const size_t kIdxIncomeTaxOffset = kGlobalStateSize - 3;
const size_t kIdxCorporateTaxOffset = kGlobalStateSize - 2;
const size_t kIdxTimeOffset = kGlobalStateSize - 1;

// Consumer actions: consume, work, choose which firm to work for
const size_t kActionSizeConsumer = kNumFirms + 1 + 1;

// add budget and theta
const size_t kStateSizeConsumer = kGlobalStateSize + 1 + 1;
const size_t kIdxConsumerBudgetOffset = 0;

// offset from agent-specific state part of array
const size_t kIdxConsumerThetaOffset = 1;

// UNUSED for consumer. Actions are floats.
// __constant__ float cs_index_to_action[num_actions_consumer *
// kActionSizeConsumer]; const size_t kActionSizeConsumer = kNumFirms +
// kNumFirms; // consume + work
/*const size_t num_actions_consumer =
    NUMACTIONSkConsumerType; // depends on discretization*/

// Firm actions: set wage, set price, invest in capital
const size_t kActionSizeFirm = 3;

// Number of actions depends on discretization of continuous action space.
const size_t kNumActionsFirm = M_NUMACTIONSFIRM;

// budget, capital, production alpha, and one-hot firm ID
const size_t kStateSizeFirm = kGlobalStateSize + 1 + 1 + 1 + kNumFirms;

// offset from agent-specific state part of array
const size_t kIdxFirmBudgetOffset = 0;
const size_t kIdxFirmCapitalOffset = 1;
const size_t kIdxFirmAlphaOffset = 2;
const size_t kIdxFirmOnehotOffset = 3;

// Constant memory available from ALL threads.
// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constant
__constant__ float kFirmIndexToAction[kNumActionsFirm * kActionSizeFirm];

// Corporate + income tax rates
const size_t kGovtActionSize = 2;
const size_t kNumActionsGovernment = M_NUMACTIONSGOVERNMENT;
const size_t kGovtStateSize = kGlobalStateSize;
__constant__ float
    kGovernmentIndexToAction[kNumActionsGovernment * kGovtActionSize];

// One RNG state for each thread. Each thread is assigned to an agent in an env.
__device__ curandState_t
    *rng_state_arr[kBatchSize * kNumAgents]; // not sure best way to do this

// Offsets into action vectors
const size_t kIdxConsumerDemandedOffset = 0;
const size_t kIdxConsumerWorkedOffset = kNumFirms;
const size_t kIdxConsumerWhichFirmOffset = kNumFirms + 1;

// currently 1 govt
const size_t kIdxThisThreadGovtId = 0;

extern "C" {

// ------------------
// CUDA C Utilities
// ------------------
__device__ void CopyFloatArraySlice(float *start_point, int num_elems,
                                    float *destination) {
  for (int i = 0; i < num_elems; i++) {
    destination[i] = start_point[i];
  }
}

__device__ void CopyIntArraySlice(int *start_point, int num_elems,
                                  float *destination) {
  for (int i = 0; i < num_elems; i++) {
    destination[i] = start_point[i];
  }
}

// unfortunately, you can't do templates with extern "C" linkage required for
// CUDA, so we have to define different functions for each case.
__device__ int *GetPointerFromMultiIndexFor3DIntTensor(int *array,
                                                       const dim3 &sizes,
                                                       const dim3 &index) {
  unsigned int flat_index =
      index.z + index.y * (sizes.z) + index.x * (sizes.z * sizes.y);
  return &(array[flat_index]);
}

__device__ float *GetPointerFromMultiIndexFor3DFloatTensor(float *array,
                                                           const dim3 &sizes,
                                                           const dim3 &index) {
  unsigned int flat_index =
      index.z + index.y * (sizes.z) + index.x * (sizes.z * sizes.y);
  return &(array[flat_index]);
}

__device__ float *
GetPointerFromMultiIndexFor4DTensor(float *array, const size_t *sizes,
                                    const size_t *multi_index) {
  // don't use this for arrays that arne't exactly size 4!!!
  unsigned int flat_index = multi_index[3] + multi_index[2] * sizes[3] +
                            multi_index[1] * sizes[3] * sizes[2] +
                            multi_index[0] * sizes[3] * sizes[2] * sizes[1];
  return &(array[flat_index]);
}

__global__ void CudaInitKernel(int seed) {
  // we want to reset random seeds for all firms and consumers
  int tidx = threadIdx.x;
  const int kThisThreadGlobalArrayIdx = blockIdx.x * kNumAgents + threadIdx.x;

  if (tidx < kNumAgents) {
    curandState_t *s = new curandState_t;
    if (s != 0) {
      curand_init(seed, kThisThreadGlobalArrayIdx, 0, s);
    }
    rng_state_arr[kThisThreadGlobalArrayIdx] = s;
  }
}

__global__ void CudaFreeRand() {
  int tidx = threadIdx.x;
  const int kThisThreadGlobalArrayIdx = blockIdx.x * kNumAgents + threadIdx.x;

  if (tidx < kNumAgents) {
    curandState_t *s = rng_state_arr[kThisThreadGlobalArrayIdx];
    delete s;
  }
}

__device__ int SearchIndex(float *distr, float p, int l, int r) {
  int mid;
  int left = l;
  int right = r;

  while (left <= right) {
    mid = left + (right - left) / 2;
    if (distr[mid] == p) {
      return mid;
    } else if (distr[mid] < p) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return left > r ? r : left;
}

// --------------------
// Simulation Utilities
// --------------------
__device__ AgentType GetAgentType(const int agent_id) {
  if (agent_id < kNumConsumers) {
    return kConsumerType;
  } else if (agent_id < (kNumConsumers + kNumFirms)) {
    return kFirmType;
  } else {
    return kGovernmentType;
  }
}

__device__ float GetCRRAUtil(float consumption, float crra_param) {
  return (powf(consumption + 1, 1.0 - crra_param) - 1.0) / (1.0 - crra_param);
}

__global__ void CudaResetEnv(float *cs_state_arr, float *fm_state_arr,
                             float *govt_state_arr, float *cs_state_ckpt_arr,
                             float *fm_state_ckpt_arr,
                             float *govt_state_ckpt_arr,
                             float theta_anneal_factor) {
  /*
  Resets the environment by writing the initial state (checkpoint) into the
  state array for this agent (thread).
  */

  const int kBlockId = blockIdx.x;
  const int kWithinBlockAgentId = threadIdx.x;

  if (kWithinBlockAgentId >= kNumAgents) {
    return;
  }

  AgentType ThisThreadAgentType = GetAgentType(kWithinBlockAgentId);

  float *state_arr;
  float *ckpt_arr;
  dim3 my_state_shape, my_state_idx;
  size_t my_state_size;

  if (ThisThreadAgentType == kConsumerType) {
    // This thread/agent is a consumer.
    my_state_size = kStateSizeConsumer;
    my_state_shape = {kBatchSize, kNumConsumers, kStateSizeConsumer};
    my_state_idx = {kBlockId, kWithinBlockAgentId, 0};
    state_arr = cs_state_arr;
    ckpt_arr = cs_state_ckpt_arr;
  } else if (ThisThreadAgentType == kFirmType) {
    // This thread/agent is a firm.
    my_state_size = kStateSizeFirm;
    my_state_shape = {kBatchSize, kNumFirms, kStateSizeFirm};
    my_state_idx = {kBlockId,
                    (unsigned int)(kWithinBlockAgentId - kNumConsumers), 0};
    state_arr = fm_state_arr;
    ckpt_arr = fm_state_ckpt_arr;
  } else {
    // This thread/agent is government.
    my_state_size = kGovtStateSize;
    my_state_shape = {kBatchSize, kNumGovts, kGovtStateSize};
    my_state_idx = {
        kBlockId,
        (unsigned int)(kWithinBlockAgentId - kNumConsumers - kNumFirms), 0};
    state_arr = govt_state_arr;
    ckpt_arr = govt_state_ckpt_arr;
  }

  float *my_state_arr = GetPointerFromMultiIndexFor3DFloatTensor(
      state_arr, my_state_shape, my_state_idx);

  float *my_ckpt_ptr = GetPointerFromMultiIndexFor3DFloatTensor(
      ckpt_arr, my_state_shape, my_state_idx);

  CopyFloatArraySlice(my_ckpt_ptr, my_state_size, my_state_arr);

  // anneal theta
  if (ThisThreadAgentType == kConsumerType) {
    my_state_arr[kGlobalStateSize + kIdxConsumerThetaOffset] *=
        theta_anneal_factor;
  }
}

__device__ void GetAction(float *action_arr, float *index_to_action_arr,
                          int index, int agent_idx, int agent_action_size) {

  // it needs to be possible to call this for either agents or firms
  // Note: each thread is an agent.
  for (int i = 0; i < agent_action_size; i++) {
    action_arr[agent_idx * agent_action_size + i] =
        index_to_action_arr[index * agent_action_size + i];
  }
}

__global__ void CudaSampleFirmAndGovernmentActions(
    float *fm_distr, int *fm_action_indices_arr, float *fm_actions_arr,
    float *govt_distr, int *govt_action_indices_arr, float *govt_actions_arr) {
  // Samples actions for firms and governments. Consumer actions are sampled in
  // Pytorch...
  const int kWithinBlockAgentId = threadIdx.x;

  // Unused threads should not do anything.
  if (threadIdx.x >= kNumAgents) {
    return;
  }

  AgentType ThisThreadAgentType = GetAgentType(kWithinBlockAgentId);

  // Index into rand states array
  int kThisThreadGlobalArrayIdx = blockIdx.x * kNumAgents + threadIdx.x;
  curandState_t rng_state = *rng_state_arr[kThisThreadGlobalArrayIdx];
  *rng_state_arr[kThisThreadGlobalArrayIdx] = rng_state;

  // float cs_cum_dist[num_actions_consumer];
  float fm_cum_dist[kNumActionsFirm];
  float govt_cum_dist[kNumActionsGovernment];

  float *my_cumul_dist;
  float *my_dist;
  int *my_indices;
  float *my_actions;
  float *index_to_action;
  int this_thread_global_array_idx;
  size_t my_num_actions;
  int my_action_size;

  // Consumers have multiple action heads, hence sampling is more complicated.
  if (ThisThreadAgentType == kConsumerType) {
    return;
  } else if (ThisThreadAgentType == kFirmType) {
    // on firm thread
    my_cumul_dist = fm_cum_dist;
    my_dist = fm_distr;
    my_indices = fm_action_indices_arr;
    my_actions = fm_actions_arr;
    my_num_actions = kNumActionsFirm;
    index_to_action = kFirmIndexToAction;
    my_action_size = kActionSizeFirm;
    this_thread_global_array_idx =
        (blockIdx.x * kNumFirms) + (threadIdx.x - kNumConsumers);
  } else {
    my_cumul_dist = govt_cum_dist;
    my_dist = govt_distr;
    my_indices = govt_action_indices_arr;
    my_actions = govt_actions_arr;
    my_num_actions = kNumActionsGovernment;
    index_to_action = kGovernmentIndexToAction;
    my_action_size = kGovtActionSize;
    this_thread_global_array_idx =
        (blockIdx.x * kNumGovts) + (threadIdx.x - kNumConsumers - kNumFirms);
  }

  // Compute CDF
  my_cumul_dist[0] = my_dist[this_thread_global_array_idx * my_num_actions];
  for (int i = 1; i < my_num_actions; i++) {
    my_cumul_dist[i] =
        my_dist[this_thread_global_array_idx * my_num_actions + i] +
        my_cumul_dist[i - 1];
  }

  // Given sampled action which is a float in [0, 1], find the corresponding
  // discrete action.
  float sampled_float = curand_uniform(&rng_state);
  const int index =
      SearchIndex(my_cumul_dist, sampled_float, 0, (int)(my_num_actions - 1));
  my_indices[this_thread_global_array_idx] = index;
  GetAction(my_actions, index_to_action, index, this_thread_global_array_idx,
            my_action_size);
}

__device__ float GetFirmProduction(float technology, float capital, float hours,
                                   float alpha) {
  if (hours < kLaborFloor) {
      hours = 0.0;
  }
  return technology * powf(capital, 1.0 - alpha) * powf(hours, alpha);
}

// --------------------
// Simulation Logic
// --------------------
__global__ void
CudaStep(float *cs_state_arr, float *cs_actions_arr, float *cs_rewards_arr,
         float *cs_state_arr_batch, float *cs_rewards_arr_batch,

         float *fm_state_arr, int *fm_action_indices_arr, float *fm_actions_arr,
         float *fm_rewards_arr, float *fm_state_arr_batch,
         int *fm_actions_arr_batch, float *fm_rewards_arr_batch,

         float *govt_state_arr, int *govt_action_indices_arr,
         float *govt_actions_arr, float *govt_rewards_arr,
         float *govt_state_arr_batch, int *govt_actions_arr_batch,
         float *govt_rewards_arr_batch,
         float *consumer_aux_batch,
         float *firm_aux_batch,
         int iter) {
  // This function should be called with 1 block per copy of the environment.
  // Within a block, each agent should have a thread.
  const int kWithinBlockAgentId = threadIdx.x;

  // return if we're on an extra thread not corresponding to an agent
  if (kWithinBlockAgentId >= kNumAgents) {
    return;
  }

  // -------------------------------------
  // Start of variables and pointers defs.
  // -------------------------------------

  // __shared__ variables are block-local: can be seen by each thread ** in the
  // block **
  __shared__ float gross_demand_arr[kNumFirms];
  __shared__ int num_consumer_demand_arr[kNumFirms];
  __shared__ float hours_worked_arr[kNumFirms];
  __shared__ float total_actually_consumer_arr[kNumFirms];
  __shared__ float bought_by_importer_arr[kNumFirms];
  __shared__ float next_global_state_arr[kGlobalStateSize];
  __shared__ float tax_revenue_arr[kNumGovts];
  __shared__ float total_utility_arr[kNumGovts];
  __shared__ bool need_to_ration_this_good_arr[kNumFirms]; // whether or not to
                                                           // ration good i

  float net_demand_arr[kNumFirms]; // amount demanded after budget constraints
                                   // by a consumer (ignore for non-consumers)

  int num_iter = (int)kMaxTime;
  AgentType ThisThreadAgentType = GetAgentType(kWithinBlockAgentId);
  float this_agent_reward = 0.0;

  // pointer to start of state vector
  // state vector consists of global state, then
  // agent-specific state global part is of same size for
  // all agents, but needs to be sliced out of different
  // arrays depending on agent type
  float *my_global_state_ptr;

  float *my_action_arr;

  // pointer to start of state vector in batch history
  float *batch_state_ptr;

  // sizes and indices for strided array access
  dim3 my_state_shape, my_state_idx, action_shape;

  // shape for batched array of scalars (action ind and reward)
  dim3 batch_single_shape, single_idx;

  // pointer to action index
  int *batch_action_value_ptr;

  // pointer to batch reward
  float *batch_reward_value_ptr;

  // pointers to action index for current arrays
  int *my_action_value_ptr;

  // pointers to reward index for current arrays
  float *my_reward_value_ptr;

  float *my_aux_batch_ptr;

  // -----------------------------------
  // End of variables and pointers defs.
  // -----------------------------------

  if (ThisThreadAgentType == kConsumerType) {
    // get current state
    my_state_shape = {kBatchSize, kNumConsumers, kStateSizeConsumer};
    my_state_idx = {blockIdx.x, threadIdx.x, 0};
    my_global_state_ptr = GetPointerFromMultiIndexFor3DFloatTensor(
        cs_state_arr, my_state_shape, my_state_idx); // index)

    // get current action
    action_shape = {kBatchSize, kNumConsumers, kActionSizeConsumer};
    my_action_arr = GetPointerFromMultiIndexFor3DFloatTensor(
        cs_actions_arr, action_shape, my_state_idx);

    // index into the episode history and save prev state into it
    size_t my_batch_state_shape[] = {kBatchSize, num_iter, kNumConsumers,
                                     kStateSizeConsumer};
    size_t my_batch_state_idx[] = {blockIdx.x, iter, threadIdx.x, 0};
    batch_state_ptr = GetPointerFromMultiIndexFor4DTensor(
        cs_state_arr_batch, my_batch_state_shape, my_batch_state_idx);
    CopyFloatArraySlice(my_global_state_ptr, kStateSizeConsumer,
                        batch_state_ptr);

    size_t my_aux_batch_shape[] = {kBatchSize, num_iter, kNumConsumers, kNumFirms};
    my_aux_batch_ptr = GetPointerFromMultiIndexFor4DTensor(
        consumer_aux_batch, my_aux_batch_shape, my_batch_state_idx
    );


    // Extract pointers to rewards, batch and current
    batch_single_shape = {kBatchSize, (unsigned int)num_iter, kNumConsumers};
    single_idx = {blockIdx.x, (unsigned int)iter, threadIdx.x};

    batch_reward_value_ptr = GetPointerFromMultiIndexFor3DFloatTensor(
        cs_rewards_arr_batch, batch_single_shape, single_idx);

    my_reward_value_ptr =
        &(cs_rewards_arr[blockIdx.x * kNumConsumers + threadIdx.x]);
  }

  if (ThisThreadAgentType == kFirmType) {
    // get current state
    size_t this_thread_firm_id = (threadIdx.x - kNumConsumers);
    my_state_shape = {kBatchSize, kNumFirms, kStateSizeFirm};
    my_state_idx = {blockIdx.x, (unsigned int)this_thread_firm_id, 0};
    my_global_state_ptr = GetPointerFromMultiIndexFor3DFloatTensor(
        fm_state_arr, my_state_shape, my_state_idx);

    // get current action
    action_shape = {kBatchSize, kNumFirms, kActionSizeFirm};
    my_action_arr = GetPointerFromMultiIndexFor3DFloatTensor(
        fm_actions_arr, action_shape, my_state_idx);

    // index into the episode history and save prev state into it
    size_t my_batch_state_shape[] = {kBatchSize, num_iter, kNumFirms,
                                     kStateSizeFirm};
    size_t my_batch_state_idx[] = {blockIdx.x, iter, this_thread_firm_id, 0};
    batch_state_ptr = GetPointerFromMultiIndexFor4DTensor(
        fm_state_arr_batch, my_batch_state_shape, my_batch_state_idx);
    CopyFloatArraySlice(my_global_state_ptr, kStateSizeFirm, batch_state_ptr);

    dim3 my_aux_batch_shape = {kBatchSize, num_iter, kNumFirms};
    dim3 aux_batch_idx = {blockIdx.x, iter, this_thread_firm_id};
    my_aux_batch_ptr = GetPointerFromMultiIndexFor3DFloatTensor(
        firm_aux_batch, my_aux_batch_shape, aux_batch_idx
    );
    // extract pointers to action indices and rewards, batch and current
    batch_single_shape = {kBatchSize, (unsigned int)num_iter, kNumFirms};
    single_idx = {blockIdx.x, (unsigned int)iter,
                  (unsigned int)this_thread_firm_id};
    batch_action_value_ptr = GetPointerFromMultiIndexFor3DIntTensor(
        fm_actions_arr_batch, batch_single_shape, single_idx);
    batch_reward_value_ptr = GetPointerFromMultiIndexFor3DFloatTensor(
        fm_rewards_arr_batch, batch_single_shape, single_idx);

    const int kThisThreadFirmIdx = blockIdx.x * kNumFirms + this_thread_firm_id;
    my_action_value_ptr = &(fm_action_indices_arr[kThisThreadFirmIdx]);
    my_reward_value_ptr = &(fm_rewards_arr[kThisThreadFirmIdx]);
  }

  if (ThisThreadAgentType == kGovernmentType) {

    int this_thread_govt_id = (threadIdx.x - kNumConsumers - kNumFirms);

    my_state_shape = {kBatchSize, kNumGovts, kGovtStateSize};
    my_state_idx = {blockIdx.x, (unsigned int)this_thread_govt_id, 0};
    my_global_state_ptr = GetPointerFromMultiIndexFor3DFloatTensor(
        govt_state_arr, my_state_shape, my_state_idx); // index)

    // get current action
    action_shape = {kBatchSize, kNumGovts, kGovtActionSize};
    my_action_arr = GetPointerFromMultiIndexFor3DFloatTensor(
        govt_actions_arr, action_shape, my_state_idx);

    // index into the episode history and save prev state into it
    size_t my_batch_state_shape[] = {kBatchSize, num_iter, kNumGovts,
                                     kGovtStateSize};
    size_t my_batch_state_idx[] = {blockIdx.x, iter, this_thread_govt_id, 0};
    batch_state_ptr = GetPointerFromMultiIndexFor4DTensor(
        govt_state_arr_batch, my_batch_state_shape, my_batch_state_idx);
    CopyFloatArraySlice(my_global_state_ptr, kGovtStateSize, batch_state_ptr);

    // extract pointers to action indices and rewards, batch and current
    batch_single_shape = {kBatchSize, (unsigned int)num_iter, kNumGovts};
    single_idx = {blockIdx.x, (unsigned int)iter,
                  (unsigned int)this_thread_govt_id};
    batch_action_value_ptr = GetPointerFromMultiIndexFor3DIntTensor(
        govt_actions_arr_batch, batch_single_shape, single_idx);
    batch_reward_value_ptr = GetPointerFromMultiIndexFor3DFloatTensor(
        govt_rewards_arr_batch, batch_single_shape, single_idx);

    const int kThisThreadGovtIdx = blockIdx.x * kNumGovts + this_thread_govt_id;
    my_action_value_ptr = &(govt_action_indices_arr[kThisThreadGovtIdx]);
    my_reward_value_ptr = &(govt_rewards_arr[kThisThreadGovtIdx]);
  }

  // ----------------------------------------------
  // State pointers and variables
  // Create pointers to agent-specific state that will be updated by the
  // simulation logic.
  // ----------------------------------------------
  float *my_state_arr = &(my_global_state_ptr[kGlobalStateSize]);
  float *prices_arr = &(my_global_state_ptr[kIdxPricesOffset]);
  float *wages_arr = &(my_global_state_ptr[kIdxWagesOffset]);
  float *available_stock_arr = &(my_global_state_ptr[kIdxStockOffset]);
  float time = my_global_state_ptr[kIdxTimeOffset];
  float income_tax_rate = my_global_state_ptr[kIdxIncomeTaxOffset];
  float corporate_tax_rate = my_global_state_ptr[kIdxCorporateTaxOffset];

  // -------------------------------
  // Safely initialize shared memory
  // -------------------------------
  if (ThisThreadAgentType == kFirmType) {
    int this_thread_firm_id = threadIdx.x - kNumConsumers;
    gross_demand_arr[this_thread_firm_id] = 0.0;
    num_consumer_demand_arr[this_thread_firm_id] = 0;
    hours_worked_arr[this_thread_firm_id] = 0.0;
    total_actually_consumer_arr[this_thread_firm_id] = 0.0;
    need_to_ration_this_good_arr[this_thread_firm_id] = false;
  }

  if (ThisThreadAgentType == kGovernmentType) {
    tax_revenue_arr[0] = 0.0;
  }

  __syncthreads();
  // -------------------------------------
  // End - Safely initialize shared memory
  // -------------------------------------

  // -------------------------------------
  // Process actions
  // -------------------------------------
  if (ThisThreadAgentType == kConsumerType) {
    // amount demanded is just the first part of the action vector
    float *this_agent_gross_demand_arr =
        &(my_action_arr[kIdxConsumerDemandedOffset]);
    const float this_agent_hours_worked =
        my_action_arr[kIdxConsumerWorkedOffset];
    const int worked_for_this_firm_id =
        (int)my_action_arr[kIdxConsumerWhichFirmOffset];

    // here, need to scale demands to meet the budget. put them in a local array
    // *budgetDemanded logic should be: compute total expenditure given prices.
    // if less than budget, copy existing demands
    float __cost_of_demand = 0.0;
    for (int i = 0; i < kNumFirms; i++) {
      __cost_of_demand += this_agent_gross_demand_arr[i] * prices_arr[i];
    }

    // Scale demand to ensure that total demand at most equals total supply
    // we want: my_state_arr being 0 always sends __scale_factor to 0
    float __scale_factor = 1.0;

    if ((__cost_of_demand > 0.0) && (__cost_of_demand > my_state_arr[0])) {
      __scale_factor = my_state_arr[0] / __cost_of_demand;
    }

    // otherwise scale all demands down to meet budget
    // copy them into a demanded array
    for (int i = 0; i < kNumFirms; i++) {
      net_demand_arr[i] = __scale_factor * this_agent_gross_demand_arr[i];
    }

    // adding up demand across threads **in the block**
    // somehow store amount demanded per firm in an array demanded (copy from
    // action) also store amount worked per firm in array worked

    // Every thread executes atomicAdd_block in a memory-safe way.
    for (int i = 0; i < kNumFirms; i++) {
      // sum across threads in block
      atomicAdd_block(&(gross_demand_arr[i]), net_demand_arr[i]);

      // increment count of consumers who want good i
      if (net_demand_arr[i] > 0) {
        atomicAdd_block(&(num_consumer_demand_arr[i]), 1);
      }
    }

    // increment total hours worked for firm i
    atomicAdd_block(&(hours_worked_arr[worked_for_this_firm_id]),
                    this_agent_hours_worked);
  }

  // wait for everyone to finish tallying up their adding
  __syncthreads();

  if (ThisThreadAgentType == kFirmType) {
    // check each firm if rationing needed
    int this_thread_firm_id = threadIdx.x - kNumConsumers;
    need_to_ration_this_good_arr[this_thread_firm_id] =
        ((gross_demand_arr[this_thread_firm_id] > 0.0) && (gross_demand_arr[this_thread_firm_id] >
         available_stock_arr[this_thread_firm_id]));
  }

  // wait for single thread to finish checking demands
  __syncthreads();

  // ----------------------------------------
  // Consumers: Rationing demand + Utility
  // ----------------------------------------
  // Logic:
  // case 1: no overdemand
  // case 2: overdemand, but some want less than 1/N -- fill everyone up to
  // max(theirs, 1/N) case 3: overdemand, everyone wants more -- fill everyone
  // up to max(theirs, 1/N)
  float net_consumed_arr[kNumFirms]; // per consumer thread
  // always add negligible positive money to avoid budgets becoming small
  // negative numbers otherwise, when computing proportions, one may end up with
  // negative stocks.
  float cs_budget_delta = 0.01;
  float fm_budget_delta = 0.01;
  float capital_delta = 0.0;

  if (ThisThreadAgentType == kConsumerType) {
    // find out how much consumed
    for (int i = 0; i < kNumFirms; i++) {
      float __ration_factor = 1.0;

      if (need_to_ration_this_good_arr[i]) {
        // overdemanded
        __ration_factor = available_stock_arr[i] / gross_demand_arr[i];
      }
      
      net_consumed_arr[i] = __ration_factor * net_demand_arr[i];

      atomicAdd_block(&(total_actually_consumer_arr[i]), net_consumed_arr[i]);
    }

    // store amount actually consumed for this consumer
    CopyFloatArraySlice(net_consumed_arr, kNumFirms, my_aux_batch_ptr);

    // ----------------------------------------
    // Compute consumer utility
    // ----------------------------------------
    float hours_worked = my_action_arr[kIdxConsumerWorkedOffset];
    int worked_for_this_firm_id =
        (int)my_action_arr[kIdxConsumerWhichFirmOffset];

    // budget is first elem of consumer state, theta second
    float __theta = my_state_arr[1];

    float __this_consumer_util = 0.0;
    float __total_hours_worked = 0.0;
    float __gross_income = 0.0;

    // Compute expenses
    // Each consumer can consume from each firm, so loop over them.
    for (int i = 0; i < kNumFirms; i++) {
      __this_consumer_util += GetCRRAUtil(net_consumed_arr[i], kCrraParam);
      cs_budget_delta -= prices_arr[i] * net_consumed_arr[i];
    }

    // Compute income
    __total_hours_worked += hours_worked;
    __gross_income += wages_arr[worked_for_this_firm_id] * hours_worked;
    float __income_tax_paid = income_tax_rate * __gross_income;
    cs_budget_delta += (__gross_income - __income_tax_paid);

    // Update tax revenue (government)
    atomicAdd_block(&(tax_revenue_arr[kIdxThisThreadGovtId]),
                    __income_tax_paid);

    // Compute reward
    this_agent_reward +=
        __this_consumer_util - (__theta / 2.0) * (__total_hours_worked);
  }

  __syncthreads();


  // ----------------------------------------
  // Firms Exports: Add external consumption.
  // ----------------------------------------
  if (ThisThreadAgentType == kFirmType ) {
      const int this_thread_firm_id = threadIdx.x - kNumConsumers;
      if (kUseImporter) {
          // sell remaining goods, if any, to importer, if price is high enough.
          float __this_firm_price = prices_arr[this_thread_firm_id];
          float __stock_after_consumers = available_stock_arr[this_thread_firm_id] - total_actually_consumer_arr[this_thread_firm_id];

          if (__this_firm_price >= kImporterPrice) {
              bought_by_importer_arr[this_thread_firm_id] = fmaxf(fminf(__stock_after_consumers, kImporterQuantity), 0.0); // floor to zero to avoid small negative floats
          }
          else {
              bought_by_importer_arr[this_thread_firm_id] = 0.0;
          }
      }
      else {
          bought_by_importer_arr[this_thread_firm_id] = 0.0;
      }
  }

  // ----------------------------------------
  // Firms: Rationing demand + Utility
  // ----------------------------------------
  if (ThisThreadAgentType == kFirmType) {
    const int this_thread_firm_id = threadIdx.x - kNumConsumers;

    float __this_firm_revenue =
        (total_actually_consumer_arr[this_thread_firm_id] + bought_by_importer_arr[this_thread_firm_id]) *
        prices_arr[this_thread_firm_id];
    float __wages_paid =
        hours_worked_arr[this_thread_firm_id] * wages_arr[this_thread_firm_id];

    // Firms can invest in new capital. This increases their production factor
    // (see GetFirmProduction).
    // here, after consumers consume, if price is >= than importer price, importer consumes up to their maximum of the goods, at the importer price

    float __gross_income = __this_firm_revenue - __wages_paid;
    capital_delta = fmaxf(my_action_arr[2] * __gross_income, 0.0);
    float __gross_profit = __gross_income - capital_delta;
    float __corp_tax_paid = corporate_tax_rate * fmaxf(__gross_profit, 0.0);
    fm_budget_delta = (__gross_profit - __corp_tax_paid);
    if (kIncentivizeFirmActivity) {
       if ((fm_budget_delta + my_state_arr[0]) > 0.0) {  // if positive budget
            this_agent_reward += (kFirmBoostRewardFactor * __this_firm_revenue);
       }
    }
    this_agent_reward += (__gross_profit - __corp_tax_paid);

    atomicAdd_block(&(tax_revenue_arr[0]), __corp_tax_paid);

    float __production = GetFirmProduction(0.01, my_state_arr[kIdxFirmCapitalOffset],
                          hours_worked_arr[this_thread_firm_id],  my_state_arr[kIdxFirmAlphaOffset]);

    // -------------------
    // Update global state
    // -------------------
    // update prices in global state
    next_global_state_arr[kIdxPricesOffset + this_thread_firm_id] =
        my_action_arr[0];
    // update wages in global state
    next_global_state_arr[kIdxWagesOffset + this_thread_firm_id] =
        my_action_arr[1];
    // update stocks in global state
    next_global_state_arr[kIdxStockOffset + this_thread_firm_id] =
        available_stock_arr[this_thread_firm_id] -
        total_actually_consumer_arr[this_thread_firm_id] - bought_by_importer_arr[this_thread_firm_id] + __production;

    *my_aux_batch_ptr = bought_by_importer_arr[this_thread_firm_id];

    // update overdemanded in global state
    next_global_state_arr[kIdxOverdemandOffset + this_thread_firm_id] =
        need_to_ration_this_good_arr[this_thread_firm_id] ? 1.0 : 0.0;
  }

  // -----------------
  // Move time forward
  // -----------------
  // Let first firm tick time
  if (ThisThreadAgentType == kFirmType) {
      const int this_thread_firm_id = threadIdx.x - kNumConsumers;
    if (this_thread_firm_id == 0) {
      next_global_state_arr[kIdxTimeOffset] =
          my_global_state_ptr[kIdxTimeOffset] + 1.0;
    }
  }

  __syncthreads();

  // ----------------------------------------
  // Subsidies
  // ----------------------------------------
  // need to redistribute tax revenues
  if (ThisThreadAgentType == kConsumerType) {
    float __redistribution = tax_revenue_arr[0] / kNumConsumers;
    cs_budget_delta += __redistribution;
  }

  // ----------------------------------------
  // Social welfare
  // ----------------------------------------
  // After this point consumers and firms know their final reward, so can inform
  // the government thread via shared memory

  __syncthreads();

  // ----------------------------------------
  // Government sets taxes for the next round
  // ----------------------------------------
  if (ThisThreadAgentType == kGovernmentType) {
    next_global_state_arr[kIdxIncomeTaxOffset] = my_action_arr[0];
    next_global_state_arr[kIdxCorporateTaxOffset] = my_action_arr[1];
  }

  __syncthreads();

  // -----------------------------------------------
  // Copy next_global_state_arr into my global state
  // -----------------------------------------------
  // All agents need to see the updated global state
  CopyFloatArraySlice(next_global_state_arr, kGlobalStateSize,
                      my_global_state_ptr);

  // -----------------------------------------------
  // Update budgets
  // -----------------------------------------------
  // Update budget (same for all agents)
  if (ThisThreadAgentType == kConsumerType) {
    my_state_arr[0] += cs_budget_delta;
  }
  if (ThisThreadAgentType == kFirmType) {
    my_state_arr[0] += fm_budget_delta;
  }

  // Add interest rate on savings
  if ((ThisThreadAgentType == kConsumerType) ||
      (ThisThreadAgentType == kFirmType)) {
    if (my_state_arr[0] > 0.0) {
      my_state_arr[0] += my_state_arr[0] * kInterestRate;
    }
  }

  // Add new capital
  if (ThisThreadAgentType == kFirmType) {
    my_state_arr[kIdxFirmCapitalOffset] += capital_delta;
  }

  // Add new capital
  if ((ThisThreadAgentType == kFirmType) ||
      (ThisThreadAgentType == kGovernmentType)) {
    *batch_action_value_ptr = *my_action_value_ptr;
  }

  // Update rewards in global state
  *my_reward_value_ptr = this_agent_reward;
  *batch_reward_value_ptr = this_agent_reward;
}

// ************************
// End of extern "C" block.
// ************************
}
