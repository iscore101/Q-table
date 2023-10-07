import numpy as np
import random
#make sure max_input_rate is divisible by num_partitions
#we partition the max_input_rate into num_partitions number of intervals

#state: tuple: (input_rate, dictionary of parralelisms)
#action: dictionary of parralelisms

import copy
class q_learner:
    def __init__(self, num_operators, max_parralelism, num_partitions_input, max_input_rate, num_partitions_selevtivity, num_partitions_processing, max_processing_rate, alpha, gamma, QOS_output):
        self.num_operators = num_operators
        self.max_parralelism = max_parralelism
        self.num_partitions_input = num_partitions_input
        self.max_input_rate = max_input_rate

        self.num_partitions_selectivity = num_partitions_selevtivity
        self.max_selectivity_rate = 1

        self.num_partitions_processing = num_partitions_processing
        self.max_processing_rate = max_processing_rate

        self.alpha = alpha
        self.gamma = gamma
        #use find_interval(max_input_rate, num_partitions, input_rate) to find the partition number to index
        #the max values inputted must be divisible by the num_partitions


        # Initialize Q-table, with these dimensions:
        #(input_partitions, parralelism * num_operators, selectivity * num_operators, processing_rate * num_operators, action_parralelism * num_operators)

        
        #first dimension is input_partitions
        self.Q = np.zeros(self.num_partitions_input)
        self.dimension_to_max = []
            #maps dimension (index) to its max
        #add parralelism, selectivity, processing_rate, action_parralelism which depends on num_operators
        for i in range(self.num_operators):
            #parralelism
            # Expanding the array using np.newaxis
            expanded_Q = self.Q[..., np.newaxis]
    
            # Creating an array of zeros with the desired shape
            zeros_shape = expanded_Q.shape + (self.max_parralelism - 1,)
            zeros_array = np.zeros(zeros_shape)

            #Concatenating the arrays along the new axis
            self.Q = np.concatenate([expanded_Q, zeros_array], axis=-1)
            self.dimension_to_max.append(self.max_parralelism) 
        for i in range(self.num_operators):
            #selectivity_rate
            expanded_Q = self.Q[..., np.newaxis]
    
            # Creating an array of zeros with the desired shape
            zeros_shape = expanded_Q.shape + (self.max_selectivity_rate - 1,)
            zeros_array = np.zeros(zeros_shape)

            #Concatenating the arrays along the new axis
            self.Q = np.concatenate([expanded_Q, zeros_array], axis=-1)
            self.dimension_to_max.append(self.max_selectivity_rate) 
        for i in range(self.num_operators):
            #processing_rate
            expanded_Q = self.Q[..., np.newaxis]
        
            # Creating an array of zeros with the desired shape
            zeros_shape = expanded_Q.shape + (self.max_processing_rate - 1,)
            zeros_array = np.zeros(zeros_shape)

            #Concatenating the arrays along the new axis
            self.Q = np.concatenate([expanded_Q, zeros_array], axis=-1)
            self.dimension_to_max.append(self.max_processing_rate) 
        for i in range(self.num_operators):
            #action_parralelism
            expanded_Q = self.Q[..., np.newaxis]
    
            # Creating an array of zeros with the desired shape
            zeros_shape = expanded_Q.shape + (self.max_parralelism - 1,)
            zeros_array = np.zeros(zeros_shape)

            #Concatenating the arrays along the new axis
            self.Q = np.concatenate([expanded_Q, zeros_array], axis=-1)
            self.dimension_to_max.append(self.max_parralelism) 


    def find_interval(self,max_value, num_partitions, value):
        #use find_interval outside this method when passing in states.        
        interval_length = max_value/num_partitions #max_value must be divisible by num_partitions
        for i in range(1, num_partitions+1):
            if i * interval_length > value:
                return i #value is in the i-th interval
        return "error; value > max_value"
    
    def partition_to_value(self, max_value, num_partitions, partition):
        interval_length = max_value/num_partitions
        return (partition + 0.5) * interval_length

    def partitions_to_values(self, input_array):
        curr_ind = 0
        state_action = copy.deepcopy(input_array)
        state_action[0] = self.partition_to_value(self.max_input_rate,self.num_partitions_input,state_action[0])

        #parralelism is good, so skip to curr_ind + num_operators
        curr_ind += self.num_operators
        #selectivity
        for i in range(self.num_operators):
            curr_ind += 1
            partition = state_action[curr_ind]
            state_action[curr_ind] = self.partition_to_value(self.max_selectivity_rate, self.num_partitions_selectivity, partition)
        
        #processing rate
        for i in range(self.num_operators):
            curr_ind += 1
            partition = state_action[curr_ind]
            state_action[curr_ind] = self.partition_to_value(self.max_processing_rate, self.num_partitions_processing, partition)
        #action parralelism is already a value

        return state_action


    def populate_q_table_offline(self,state):
        #generates action given a state, make sure to use find_interval
        #store the state passed in as last_state

        #we call recursive looper if we need another nested loop
        def recursive_looper(current_indexing, curr_loop):
            for i in range(self.dimension_to_max[curr_loop]):

                # Update the Q-value
                next_indexing = copy.deepcopy(current_indexing)
                next_indexing[curr_loop] = i
                #figure out how to index, based on metrics
                if curr_loop != len(self.dimension_to_max)-1:
                    recursive_looper(next_indexing, curr_loop + 1)
                else:
                    #reached last action loop, update q value.

                    #pass state with action's parralelism
                    reward = modifiedDS2(self.partitions_to_values(next_indexing))
                    #the last num_operators are action parralelism

                    #do something to reward??

                    self.set_q_table(next_indexing, reward)
                    #no recursion, update q value
        #current indexing format: list of size self.dimension to max
        current_indexing = [0 for i in range(self.dimension_to_max)]


        
        return self.Q

    def set_q_table(self, indexing, set_value):
        self.Q[*indexing] = set_value

    

    #given an action and curent state, find the next state
    def get_next_state(self,state, action):
        #assume input metrics stay the same, we are not forecasting them. change the parralelism to the action's
        
        #returns (metrics of state, action (max_parralelism and num_partition))
        for i in range(self.num_operators):
            state[i] = action[i]

        return state
        #change this accordingly to how many metrics
    
    
    
    def online_generate_action(self,state):
        #we seperate state and action (unline with indexing during offline)

        # Epsilon-greedy action selection
        action = []
        if random.uniform(0, 1) < self.epsilon:
            for i in range(self.num_operators):
                action.append(random.randint(0, self.max_parralelism))
                #random parralelisms
        else:
            #index by state, then argmax on the subarray
            subarray = self.Q[tuple(state) + (Ellipsis,)]
            flattened_index = np.argmax(subarray)

            # Convert the flattened index to multi-dimensional indices
            remaining_dim_indices = np.unravel_index(flattened_index, subarray.shape)
            action = list(remaining_dim_indices)
            

        # Get the next state and reward
        self.state = state
        self.next_state = self.get_next_state(state, action)
        #sends the action to the streaming system, which then comes back with the reward.
        execute_action(action)

    #call this after action is executed and reward is collected
    #MAKE SURE TO CALL THIS IN BETWEEN CALLS TO online_generate_action
    def online_update_q_table(self, reward):
        #we index here by doing *(state+action)
        
        #this line should index by the state, then take max over the rest of the dimensions (the action dimensions)

        best_next_action_value = np.max(self.Q[(*self.next_state, ...)]) 
        #maybe this needs to be edited to make it explicitly a tuple of unpack + ...

        value = (1 - self.alpha) * self.Q[*(self.state + self.last_action)] + self.alpha * (reward + self.gamma * best_next_action_value)            
        self.set_q_table(self.state + self.last_action, value)
        return self.Q

    #TO FIX: indexing whenever we update the q table (we need to be able to index by state, action pair), or call np.argmax
    