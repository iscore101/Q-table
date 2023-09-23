import numpy as np
import random
#make sure max_input_rate is divisible by num_partitions
#we partition the max_input_rate into num_partitions number of intervals

#state: tuple: (input_rate, dictionary of parralelisms)
#action: dictionary of parralelisms

import copy
class q_learner:
    def __init__(self, num_operators, max_parralelism, num_partitions, max_input_rate, alpha, gamma, QOS_output):
        #self.num_operators = num_operators
        self.max_parralelism = max_parralelism
        self.num_partitions = num_partitions
        self.max_input_rate = max_input_rate
        self.alpha = alpha
        self.gamma = gamma
    #use find_interval(max_input_rate, num_partitions, input_rate) to find the input rate
    #calculate the number of states (# of rows in the Q table) by taking max_parralelism ^ num_operators * num_partitions

        state_metrics_dict = {"parralelism":self.max_parralelism, "num_partition": self.num_partitions }
        #try to add max_parralelism and num_paritions last


        # Initialize Q-table, first dimension is number of partitions of the interval
        self.Q = np.zeros(num_partitions)
        self.Q_guide = [] #documents the metric of each dimension (that determines state or action)
        for j in range(2):
            for key, value in state_metrics_dict.items():
                if j == 1:
                    if key not in ["parralelism"]:
                        continue
                #action is just max_parralelism 
                
                #document the key in self.Q_guide
                if j == 0:
                    self.Q_guide.append("state_" + key)
                else:
                    self.Q_guide.append("action_" + key)
                
                # Expanding the array using np.newaxis
                expanded_Q = self.Q[..., np.newaxis]
    
                # Creating an array of zeros with the desired shape
                zeros_shape = expanded_Q.shape + (value - 1,)
                zeros_array = np.zeros(zeros_shape)

                #Concatenating the arrays along the new axis
                self.Q = np.concatenate([expanded_Q, zeros_array], axis=-1)

    def find_interval(self,max_input_rate, num_partitions, input_rate):
        #use find_interval outside this method when passing in states.        
        interval_length = max_input_rate/num_partitions #max_input_rate must be divisible by num_partitions
        for i in range(1,num_partitions+1):
            if i * interval_length > input_rate:
                return i #input_rate is in the i-th interval
        return "error; input_rate > max_input_rate"

    

    #given an action and curent state, find the next state
    def get_next_state(state, action):
        #assume input metrics stay the same, we are not forecasting them. change the parralelism to the action's
        
        #returns (metrics of state, action (max_parralelism and num_partition))
        state["parralelism"] = action["parralelism"]
        return state
        #change this accordingly to how many metrics
    
    def populate_q_table_offline(self,state):
        #generates action given a state, make sure to use find_interval
        #store the state passed in as last_state

        for i in range(self.max_parralelism):
            i += 1  
            action = {"parralelism" : i}
            next_state = self.get_next_state(state, action)
            # Update the Q-value
            reward = modifiedDS2(state, action)

            #figure out how to index, based on metrics
            best_next_action = np.argmax(self.Q[next_state])
            self.Q[self.last_state, self.last_action] = (1 - self.alpha) * self.Q[self.last_state, self.last_action] + self.alpha * (reward + self.gamma * self.Q[next_state, best_next_action])

        return self.Q
    
    def online_generate_action(self,state):
        # Epsilon-greedy action selection
        action = {}
        if random.uniform(0, 1) < self.epsilon:
            action["parralelism"] = random.randint(0, self.max_parralelism)
        else:
            action["parralelism"] = np.argmax(self.Q[state]) #get this to index by state, input the correct action index
            
        # Get the next state and reward
        self.state = state
        self.next_state = self.get_next_state(state, action)
        #sends the action to the streaming system, which then comes back with the reward.
        execute_action(action)

    #call this after action is executed and reward is collected
    #MAKE SURE TO CALL THIS IN BETWEEN CALLS TO online_generate_action
    def online_update_q_table(self, reward):
        #index correctly.
        best_next_action = np.argmax(self.Q[self.next_state])
        self.Q[self.state, self.last_action] = (1 - self.alpha) * self.Q[self.state, self.last_action] + self.alpha * (reward + self.gamma * self.Q[self.next_state, best_next_action])            
        return self.Q

    #TO FIX: indexing whenever we update the q table (we need to be able to index by state, action pair), or call np.argmax
    