import numpy as np
import random
#make sure max_input_rate is divisible by num_partitions
#we partition the max_input_rate into num_partitions number of intervals

#state: tuple: (input_rate, dictionary of parralelisms)
#action: dictionary of parralelisms

import copy
class q_learner:
    def __init__(self, num_operators, max_parralelism, num_partitions, max_input_rate, alpha, gamma, QOS_output):
        self.num_operators = num_operators
        self.max_parralelism = max_parralelism
        self.num_partitions = num_partitions
        self.max_input_rate = max_input_rate
        self.alpha = alpha
        self.gamma = gamma
        self.QOS = QOS_output
    #use find_interval(max_input_rate, num_partitions, input_rate) to find the input rate
    #calculate the number of states (# of rows in the Q table) by taking max_parralelism ^ num_operators * num_partitions
        
        #self.N = max_parralelism ** num_operators * num_partitions
        #the number of actions, A is equal to N (at any state you can go to the N-1 other states or stay)
        #self.A = self.N

        # Initialize Q-table, first dimension is number of partitions of the interval

        self.Q = np.zeros(num_partitions)
        for j in range(2):
            for i in range(self.num_operators):
                # Expanding the array using np.newaxis
                expanded_Q = self.Q[..., np.newaxis]
    
                # Creating an array of zeros with the desired shape
                zeros_shape = expanded_Q.shape + (self.max_parralelism - 1,)
                zeros_array = np.zeros(zeros_shape)

                #Concatenating the arrays along the new axis
                self.Q = np.concatenate([expanded_Q, zeros_array], axis=-1)
        #self.Q is now of shape: num_partitions, max_parralelism, max_parralelism.... num_operator times, max_parallelism.... num_operator times
        #index self.Q as follows: partition #, operator 1 parralelism, operator 2 parralelism... action op1 parralelism, action op2 parralelism...

    def find_interval(self,max_input_rate, num_partitions, input_rate):
        #use find_interval outside this method when passing in states.        
        interval_length = max_input_rate/num_partitions #max_input_rate must be divisible by num_partitions
        for i in range(1,num_partitions+1):
            if i * interval_length > input_rate:
                return i #input_rate is in the i-th interval
        return "error; input_rate > max_input_rate"

    

    #given an action and curent state, find the next state
    def get_next_state(state, action):
        #assume input rate stays the same, since we are not forecasting rate. change the parralelism to the action
        #returns (rate_interval, action's parralelism)
        return (state[0], action)
    

    
    def generate_action(self,state):
        #generates action given a state, make sure to use find_interval
        #store the state passed in as last_state

        #state format is conisdered to be dict of operator : parralelism

        self.last_state = state
        #policy to generate action at every decision point. updates last action and last state. 

        #ACTION SELECTION: DS2!

        #Loop through all possible actions to take. Pass into DS2

        starting_arr = [0] * len(self.num_operators)
        self.ret_action = None
        self.best_reward = float('-inf')
        state_arr = [state[1][i] for i in range(len(state[1]))]
        state_interval = [self.find_interval(self.max_input_rate, self.num_partitions, state[0])]
        
        def action(parralelisms_arr):
            action = {i:parralelisms_arr[i] for i in range(self.parralelisms_arr)}
            Q_index = tuple(state_interval + state_arr + parralelisms_arr)

            #call to DS2
            if self.QOS <= modified_DS2(state, action):
                if self.Q[*Q_index] > self.best_reward:
                    self.best_reward = self.Q[*Q_index]
                    self.ret_action = parralelisms_arr
                    #if it meets the QOS according to DS2 and Q-value > best Q-value, update the returned action

            #recursive to get all actions
            for i in self.num_operators:
                if parralelisms_arr[i] < self.max_parralelism:
                    new = copy.deepcopy(parralelisms_arr)
                    new[i] += 1
                    action(parralelisms_arr) 
        action(starting_arr)
        if self.ret_action == None:
            #if none of them meet QOS, simply return the best action according to DS2
            return DS2(state)

        return self.ret_action
        #returns action in array format, arr[i] is the parralelism for operator i

        #note: call this by passing in a state. use the action it returns
        #this must be called BEFORE update q table, since it updates the last state and last action


    def update_q_table(self, reward):
        #simulates a "step"
        #pass in the reward received by the system        

        #get the next_state (predicted) to update the q-table
        next_state = self.get_next_state(self.last_state, self.last_action)
            
        # Update the Q-value
        best_next_action = np.argmax(self.Q[next_state])
        self.Q[self.last_state, self.last_action] = (1 - self.alpha) * self.Q[self.last_state, self.last_action] + self.alpha * (reward + self.gamma * self.Q[next_state, best_next_action])
            
        return self.Q