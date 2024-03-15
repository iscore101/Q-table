import numpy as np
import random
from modified_ds2 import Vertex
from modified_ds2 import DirectedGraph
import copy

#make sure max_input_rate is divisible by num_partitions
#we partition the max_input_rate into num_partitions number of intervals

#state: tuple: (input_rate, dictionary of parralelisms)
#action: dictionary of parralelisms

class q_learner:

    def __init__(self, num_operators, max_parralelism, num_partitions_input, max_input_rate, num_partitions_selectivity, num_partitions_processing, max_processing_rate, alpha, gamma, epsilon, graph):
        self.num_operators = num_operators
        self.max_parralelism = max_parralelism
        self.num_partitions_input = num_partitions_input
        self.max_input_rate = max_input_rate

        self.num_partitions_selectivity = num_partitions_selectivity
        self.max_selectivity_rate = 1

        self.num_partitions_processing = num_partitions_processing
        self.max_processing_rate = max_processing_rate

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.graph = graph
        #use find_interval(max_input_rate, num_partitions, input_rate) to find the partition number to index
        #the max values inputted must be divisible by the num_partitions

        # Initialize Q-table, with these dimensions:
        #(input_rate, parralelism * num_operators, selectivity * num_operators,
        # processing_rate * num_operators, action_parralelism * num_operators)
        q_dims = [self.num_partitions_input]
        for i in range(num_operators):
            q_dims.append(self.max_parralelism)
        for i in range(num_operators):
            q_dims.append(self.num_partitions_selectivity)
        for i in range(num_operators):
            q_dims.append(self.num_partitions_processing)
        for i in range(num_operators):
            q_dims.append(self.max_parralelism)
        self.dimension_to_max_partition = copy.deepcopy(q_dims)

        self.Q = np.zeros(tuple(q_dims))
        self.state = None
        self.next_state = None
        self.last_action = None

        print("==> q_learner instantiated")


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

    def value_to_partition(self, num_partitions, max_value, value):
        partition_size = max_value / num_partitions
        partition = value // partition_size

        if partition >= num_partitions:
            partition = num_partitions - 1
        return partition

    def populate_q_table_offline(self):
        #we call recursive looper if we need another nested loop
        def recursive_looper(current_indexing, curr_loop):
            for i in range(self.dimension_to_max_partition[curr_loop]):

                # Update the Q-value
                next_indexing = copy.deepcopy(current_indexing)
                next_indexing[curr_loop] = i
                #figure out how to index, based on metrics
                if curr_loop != len(self.dimension_to_max_partition)-1:
                    recursive_looper(next_indexing, curr_loop + 1)
                else:
                    #reached last action loop, update q value.

                    #pass state with action's parralelism
                    # print("partition to vals:", self.partitions_to_values(next_indexing))
                    reward = self.graph.compute_output_rates_in_q_table(self.partitions_to_values(next_indexing))
                    #the last num_operators are action parralelism

                    #do something to reward??

                    self.Q[tuple(next_indexing)] = reward
                    #no recursion, update q value
        
        print("==> populating offline Q-table")

        #current indexing format: list of size self.dimension to max
        current_indexing = [0 for i in range(len(self.dimension_to_max_partition))]
        recursive_looper(current_indexing, 0)

        return self.Q


    #given an action and curent state, find the next state
    def get_next_state(self,state, action):
        #assume input metrics stay the same, we are not forecasting them. change the parralelism to the action's

        #returns (metrics of state, action (max_parralelism and num_partition))
        for i in range(self.num_operators):
            state[i + 1] = action[i]

        return state
        #change this accordingly to how many metrics


    def online_generate_action(self, state, ds2=False):
        print("==> generating action")
        #we seperate state and action (unlike with indexing during offline)

        ind_state = np.zeros(state.shape)
        ind_state[0] = self.value_to_partition(self.num_partitions_input, self.max_input_rate, state[0])
        for i in range(self.num_operators):
            ind_state[i + 1] = int(state[i + 1]) - 1
            ind_state[self.num_operators + (i + 1)] = self.value_to_partition(self.num_partitions_selectivity, self.max_selectivity_rate, state[self.num_operators + (i + 1)])
            ind_state[2 * self.num_operators + (i + 1)] = self.value_to_partition(self.num_partitions_processing, self.max_processing_rate, state[2 * self.num_operators + (i + 1)])

        ind_state = ind_state.astype(int)
        print(f'ind_state: {ind_state}')

        # Epsilon-greedy action selection
        if ds2:
            # override q-table stuff for raw ds2
            all_parallelisms = np.array(np.meshgrid(*[range(self.max_parralelism)] * self.num_operators)).T.reshape(-1, self.num_operators)
            input_and_parallelisms = np.concatenate((np.repeat([state[0]], len(all_parallelisms)).reshape(-1, 1), all_parallelisms), axis=1)
            all_states = np.concatenate((input_and_parallelisms, np.repeat([state[1:self.num_operators+1]], len(input_and_parallelisms), axis=0)), axis=1)
            ds2_values = np.apply_along_axis(self.graph.compute_output_rates_in_q_table, 1, all_states)
            ind_action = all_parallelisms[np.argmax(ds2_values)] # for only ds2 (would be in separate case)
        elif random.uniform(0, 1) < self.epsilon:
            ind_action = [random.randint(0, self.max_parralelism - 1) for _ in range(self.num_operators)]
        else:
            # evaluate all ds2 for lazy
            all_parallelisms = np.array(np.meshgrid(*[range(self.max_parralelism)] * self.num_operators)).T.reshape(-1, self.num_operators)
            input_and_parallelisms = np.concatenate((np.repeat([state[0]], len(all_parallelisms)).reshape(-1, 1), all_parallelisms), axis=1)
            all_states = np.concatenate((input_and_parallelisms, np.repeat([state[1:self.num_operators+1]], len(input_and_parallelisms), axis=0)), axis=1)
            ds2_values = np.apply_along_axis(self.graph.compute_output_rates_in_q_table, 1, all_states)

            #index by ind_state, then argmax on the subarray
            subarray = self.Q[tuple(ind_state)]
            # elt-wise max ds2 and subarray
            lazy_ds2 = np.where(subarray == 0, ds2_values, subarray)
            flattened_index =  np.argmax(lazy_ds2)

            # flattened_index = np.argmax(subarray)
            # Convert the flattened index to multi-dimensional indices
            remaining_dim_indices = np.unravel_index(flattened_index, subarray.shape)
            ind_action = list(remaining_dim_indices)

        print(f'==> ind_action: {ind_action}')
        # Get the next ind_state and reward
        self.state = list(ind_state)
        self.next_state = list(self.get_next_state(ind_state, ind_action))
        self.last_action = ind_action
        # sends the action to the streaming system, which then comes back with the reward.
        return [a + 1 for a in ind_action]


    #UNPACKS, ACTIONS NEED TO BE TESTED
    #call this after action is executed and reward is collected
    #MAKE SURE TO CALL THIS IN BETWEEN CALLS TO online_generate_action
    def online_update_q_table(self, reward):
        #we index here by doing *(state+action)
        print("==> updating q table")
        #this line should index by the state, then take max over the rest of the dimensions (the action dimensions)
        best_next_action_value = np.max(self.Q[tuple(self.next_state)])

        # td_error = reward + self.gamma * best_next_action_value - self.Q[tuple((self.state) + self.last_action)]
        # self.Q[tuple((self.state) + self.last_action)] += self.alpha * td_error
    
        value = (1 - self.alpha) * self.Q[tuple(self.state + self.last_action)] + self.alpha * (reward + self.gamma * best_next_action_value)
        self.Q[tuple(self.state + self.last_action)] = value

            

        return self.Q

    #TODO: indexing whenever we update the q table (we need to be able to index by state, action pair), or call np.argmax
