extern "C" {
    // Build notes - agents need to make build decision in random order
    // Cannot be parallelized
    // This level of management can be done at the env step level
    // Keep thread/block structure for now

    // Check if agent has the resources necessary to build.
    // If so, then check if tile is available.
    // Return true if all preconditions true.
    __device__ bool AgentCanBuild(
        int * agentLocations,
        float * inventory,
        const int kNumAgents,
        int * locationLandmarks,
        int * locationResources,
        const int numRows,
        const float resourceCosts[]
    ) {
        // Each agent gets a triplet of cells [Wood, Stone, Coins]
        // inventory shape = numAgents * numResources
        //
        // Ordered in triplets - e.g. inv[0], inv[1], inv[2] are the first
        // agent's inventory, inv[3], inv[4], inv[5] are the second
        // agent's inventory; so on
        //
        // Same 'chunking' as environments - that is, first 
        // numAgents * numResources items belong to the first 
        // environment, next numAgents * numResources indices to 
        // the next environment, so on so forth

        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        int timeIndependentIndexOffset = kEnvId * (kNumAgents - 1) + kAgentId;

        for(unsigned int index = 0; index < sizeof(resourceCosts); index++) {
            int indexOfInventory = timeIndependentIndexOffset + index;
            if (inventory[indexOfInventory] < resourceCosts[index]) {
                return false;
            }
        }

        int agentR = agentLocations[timeIndependentIndexOffset];
        int agentC = agentLocations[timeIndependentIndexOffset + 1];

        // locationLandmarks and locationResources are mapLen * mapLen 
        // tensors that store current map data. 
        // They are row-indexed like so:
        //
        //      [0  1  2  3]
        //      [4  5  6  7]
        //      [8  9  10  11]
        //
        // Given a coordinate pair (r, c), you can get its linear 
        // location like this:
        //
        //      lin_index = r * numRows + c
        //
        // given that r and c are indexed from the top left corner.
        // 
        // Encoding for Landmarks:
        //      0: No Landmark
        //      1: House
        //      2: Water
        //
        // Encoding for Resources:
        //      0: No Resources
        //      1: Wood
        //      2: Stone
        if (locationLandmarks[agentR * numRows + agentC] != 0) {
            return false;
        } else if (locationResources[agentR * numRows + agentC] != 0) {
            return false;
        }
        return true;
    }
    __global__ void CudaBuildStep(
        int * actions,
        int * agentLocations,
        const float laborCost,
        float * endogenous,
        int * env_timestep_arr,
        float * inventory,
        int * locationLandmarks,
        int * locationResources,
        float * agentBuildPayments,
        const float resourceCosts [],
        const int kNumAgents,
        const int kEpisodeLength,
        const int numCols,
        const int numRows,
        float * recordOfAllBuilds
    ) {
        const int kEnvId = blockIdx.x;
        const int kAgentId = threadIdx.x;

        // For baby agents only!
        if (kAgentId < (kNumAgents - 1)) { 

            // Offset for all mobiles for this specific environment
            const int kArrayIdxOffset = kEnvId * (kEpisodeLength + 1) *
                (kNumAgents - 1);

            // Find chunk of agents for this timestep, then find location
            // of this specific agent in this timestep chunk
            int time_dependent_array_index_curr_t = kArrayIdxOffset +
                env_timestep_arr[kEnvId] * (kNumAgents - 1) + kAgentId;

            // prev timestep location of this agent
            int time_dependent_array_index_prev_t = kArrayIdxOffset +
                (env_timestep_arr[kEnvId] - 1) * (kNumAgents - 1) + kAgentId;

            // index for the kth agent of this specific environment
            const int agentSpecificPointer = kEnvId * (kNumAgents - 1) +
                kAgentId;
            
            if (actions[agentSpecificPointer] == 1) {
                if (AgentCanBuild(
                    agentLocations,
                    inventory,
                    kNumAgents,
                    locationLandmarks,
                    locationResources,
                    numRows,
                    resourceCosts
                )){
                    // Remove the cost of building
                    for(unsigned int index = 0; index < sizeof(resourceCosts); index++) {
                        int indexOfInventory = 3 * agentSpecificPointer + index;
                        inventory[indexOfInventory] = inventory[indexOfInventory] - resourceCosts[index];
                    }

                    // Find location of agent
                    int agentR = agentLocations[2 * agentSpecificPointer];
                    int agentC = agentLocations[2 * agentSpecificPointer + 1];
                    
                    // Add the house, add income to inventory
                    locationLandmarks[agentR * numRows + agentC] = 1;
                    inventory[agentSpecificPointer + 2] = inventory[agentSpecificPointer + 2] 
                                                    + agentBuildPayments[agentSpecificPointer];
                    
                    // update the total labor used
                    endogenous[agentSpecificPointer] = endogenous[agentSpecificPointer] + laborCost;

                    // Logging Notes: Pass in record arrays of shape num_features * num_envs 
                    // * num_agents * num_timesteps
                    //
                    // recordOfAllBuilds indexed like so:
                    //
                    //      [row, col, payout]
                    //
                    // index(env, timestep, agent, feature, num_envs, num_timesteps, 
                    // num_agents, num_features) = (num_timesteps * num_agents * num_features) * env
                    // + (num_agents * num_features) * timestep + agent * num_features + feature
                    //
                    // Every agent receives the logging arrays
                    // Must be executed sequentially anyways

                    int recordIndex = 
                }
            }
        }
    }
}