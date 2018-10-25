from Dueling_DQN import Dueling_DQN as DDQN
import numpy as np
import random
import tensorflow as tf

class Worker(object):
    """This class is implemented to perform algorithm 1 from the paper: 
    https://arxiv.org/pdf/1602.01783.pdf

    Uses the github repository:
    https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
    as a reference.
        
    A worker object has it's copy of the environment; as it interacts with 
    the environment, it accumulates gradients which it uses to update the
    parameters of the global network. 
    
    A worker has both an online network and a target network both with dueling
    architextures. Throughout training, the online network is updated using
    targets from the target network. The weights in the online network are 
    periodically copied to the weights in the target network.
    Additionally, a worker object is able to copy the global network weights
    to its online network weights.
    
    The most important function in this class is the work() function"""

    # a static class variable that is shared among all worker instances. 
    #This approach allows epsilon to be linearly annealed across all 
    #worker objects throughout training.
    #The annealing occurs in the function anneal_epsilon() which is called
    #from the work() function
    epsilon
    epsilon_change
    num_decrease_epochs

    def __init__(self,
                input_size, #get this from len(encoded_state)
                output_size, #the number of possible actions
                architextures, #a dictionary specifying the dueling architexture
                transfer_rate, #how often to copy the online weights to target weights
                asyc_update_rate, #how often to update global network
                gamma, #discount factor
                learning_rate,
                Environment, #The copy of the environment
                start_epsilon, #for epsilon-greedy action selection
                epsilon_change, #used to anneal epsilon after every training epoch
                session, # a tensorflow session
                scope, #used to define variable scopes which are used to calculate gradients
                trainer #the algorithm used to apply gradients
                ):
        
        #target network needs to be outside of variable scope so that we don't calculate the gradients with respect to its parameters
        self.target_network = DDQN(input_size=input_size,
                                   output_size=output_size,
                                   architextures=architextures,
                                   network_name="Target")

        #initialize static epislon members
        Worker.epsilon = start_epsilon
        Worker.epsilon_change = epsilon_change

        with tf.variable_scope(scope):
            self.online_network = DDQN(input_size=input_size,
                                       output_size=output_size,
                                       architextures=architextures,
                                       network_name="Online")
            
            self.Environment=Environment
            self.gamma=gamma
            self.learning_rate=tf.constant(learning_rate, dtype=tf.float32)
            self.transfer_rate=transfer_rate
            self.asyc_update_rate=asyc_update_rate
            #self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.session=session

            #place-holders
            self.targets = tf.placeholder(dtype=tf.float32, shape=(output_size,), name="Target")

            #operations
            #mean squared error between targets and action values
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.targets, self.online_network.forward_values_graph)))
            
            #get all of the trainable network parameters
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            
            #derivatives of loss with respect to network weights 
            self.gradients = tf.gradients(self.loss, local_vars)
            
            #cast gradients to placeholders
            self.gradients_ph = []
            for tensor in self.gradients:
                self.gradients_ph.append(tf.placeholder(dtype=tf.float32, shape=tensor.shape))
            
            #global network variables    
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Global')
            
            #used to update the global network weights using accumulated gradients from work() function
            self.apply_grads = trainer.apply_gradients(zip(self.gradients_ph, global_vars))
        
        #Computation graph to copy global network weights to online network weights. 
        #Called in copy_global_weights() function
        self.update_local_ops = self.update_target_graph('Global', scope)


    def work(self, sess):
        """Implementation of algorithm 1 from the paper: https://arxiv.org/pdf/1602.01783.pdf"""
        self.copy_global_weights(sess)
        self.copy_weights()
        global T
        global T_MAX
        switch=True

        t=0
        state=self.Environment.initialize_state()
        
        while T < T_MAX:
            #Get an epsilon-greedy action
            action=self.epsilon_greedy_action(state)
            #take that action and get the reward, next_state, and terminal variables
            reward, next_state, terminal = self.Environment.take_action(action)
            #calculate the targets for training
            targets = self.get_target(state, action, reward, next_state, terminal, sess)
            #reshape the state to feed into gradient calculation
            state = np.reshape(state, (1, len(state)))
            
            if switch: #first time getting grads
                grads = sess.run(self.gradients, feed_dict={self.online_network.X_in: state,
                                                            self.targets: targets})
                switch=False
            else: #accumulate grads
                to_add=sess.run(self.gradients, feed_dict={self.online_network.X_in: state,
                                                             self.targets: targets})
                #FIXME: there might be a faster way to do this next line. This is where we're accumulating the grads
                grads=[np.array(list((map(np.add, x, y)))) for x, y in zip(grads, to_add)]

            state = next_state
            T+=1
            t+=1
            self.anneal_epsilon()
            if T % self.transfer_rate == 0: #copy online network weights to target network weights
                self.copy_weights() 
            if T % self.asyc_update_rate == 0 or terminal: #use gradients to update weights in global network
                feed_dict=dict(zip(self.gradients_ph, grads))
                sess.run(self.apply_grads, feed_dict=feed_dict)
                switch=True


    def epsilon_greedy_action(self, state):
        if random.random() < Worker.epsilon:
            return self.Environment.random_action()
        else:
            return self.greedy_action(state)
    
    def greedy_action(self, state):
        #runs the online network using the current state as input
        #returns the suggested action
        return int(self.session.run(self.online_network.forward_action_graph, 
                                    feed_dict={self.online_network.X_in: np.reshape(state, (1, self.online_network.input_size))}))

    def get_target(self, current_state, action, reward, next_state, terminal, session=None):
        """implementation of the following algorithm: https://coach.nervanasys.com/algorithms/value_optimization/double_dqn/index.html
        Calculates the target value given a (s, a, r, s', t) tuple.
        WARNING: the given algorithm does not take into account terminal transitions.
        In these instances, the played_targets should be set to the reward.
        
        Also see lines 9 and 10 in algorithm 1 from the paper https://arxiv.org/pdf/1602.01783.pdf"""
        assert session is not None, "A tf.Session() must be passed into get_targets"
        current_state = np.reshape(current_state, (1, self.online_network.input_size))
        next_state = np.reshape(next_state, (1, self.online_network.input_size))
        with tf.name_scope("Target"):
            #Using the NEXT STATE from the sampled batch, run the ONLINE network in order 
            #to find the Q maximizing action argmax_a_(Q(s', a)
            action_index = session.run(self.online_network.forward_action_graph, feed_dict={self.online_network.X_in: next_state})[0]
           
            #For this action, use the corresponding NEXT STATE and run the TARGET network 
            #to calculate the Q(s', argmax_a_(Q(s', a))
            all_t_q_vals = session.run(self.target_network.forward_values_graph, feed_dict={self.target_network.X_in: next_state})[0]
            selected_t_q_val = all_t_q_vals[action_index]

            #In order to zero out the updates for the actions that were not played 
            #(resulting from zeroing the MSE loss), use the CURRENT STATE and run the 
            #ONLINE network to get the current Q values predictions. Set those values 
            #as the targets for the actions that were not actually played.
            targets = session.run(self.online_network.forward_values_graph, feed_dict={self.online_network.X_in: current_state})[0]
            played_target = np.array(reward) + self.gamma*np.multiply(selected_t_q_val, 1-terminal)
            targets[action] = played_target
            return targets

    def update_target_graph(self, from_scope,to_scope):
        """Used to set worker network parameters to those of global network."""
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def copy_global_weights(self, session):
        """copies the global variables to the worker's online network variables"""
        session.run(self.update_local_ops)
        return

    def copy_weights(self):
        """Copies the weights from the worker's online network over to 
        the target network. Rebuilds target_network's computation graph
        Copy hidden weights and biases"""
        for i in range(len(self.online_network.hidden_weights)):
            self.target_network.hidden_weights[i]=self.online_network.hidden_weights[i]
            self.target_network.hidden_biases[i]=self.online_network.hidden_biases[i]
        #Copy value weights and biases
        for i in range(len(self.online_network.value_weights)):
            self.target_network.value_weights[i]=self.online_network.value_weights[i]
            self.target_network.value_biases[i]=self.online_network.value_biases[i]
        #Copy advantage weights and biases
        for i in range(len(self.online_network.advantage_weights)):
            self.target_network.advantage_weights[i]=self.online_network.advantage_weights[i]
            self.target_network.advantage_biases[i]=self.online_network.advantage_biases[i]
       
        #rebuild target computation graphs
        self.target_network.forward_values_graph, self.target_network.forward_action_graph = self.target_network.forward()
        return

    def anneal_epsilon(self):
        """Uses epsilon_change member to update the static member 
        Worker.epsilon.

        Changing Worker.epsilon in this way will change epsilon for all other
        worker objects as well"""
        Worker.epsilon -= abs(Worker.epsilon_change)


from pathlib import Path
import os
import platform
import sys
#Adding Agent, Environment, Experience_Replay, etc. folders to python path
sys.path.insert(0, r"C:\Users\Spencer\Documents\VisualStudio2017\Projects\KnotProblem\Knot_MDP\Agent")
sys.path.insert(0, r"C:\Users\Spencer\Documents\VisualStudio2017\Projects\KnotProblem\Knot_MDP\Environment")
from Slice_Environment_Wrapper import Slice_Environment_Wrapper as SEW
from Start_States_Buffer2 import Start_States_Buffer as SSB
import tensorflow as tf
from Worker import Worker
from Global_Network import Global_Network as GN

start_states_capacity=100000
max_braid_index=6
max_braid_length=10
seed_braids = [[1], [1, 1], [1, 1, 1]]
move_penalty=0.05

seed_prob=0.5
uniform=True

#Instantiate a Worker Object
starts_buffer=SSB(capacity=start_states_capacity,
                      max_braid_index=max_braid_index,
                      max_braid_length=max_braid_length,
                      seed_braids=seed_braids,
                      move_penalty=move_penalty)

environment_name="SliceEnv"
Environment=SEW(max_braid_index=max_braid_index,
                max_braid_length=max_braid_length,
                inaction_penalty=move_penalty,
                start_states_buffer=starts_buffer,
                seed_prob=seed_prob,
                uniform=uniform)

input_size=len(Environment.slice.encode_state())
output_size = 13
architextures={'Hidden': (512, 512, 512), 'Value': (512, 1), 'Advantage': (512, 13)}
transfer_rate=2000
asyc_update_rate=10
gamma=0.99
learning_rate=0.000000001
sess = tf.Session()

global_network = GN(input_size, output_size, architextures, scope="Global")

trainer=tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.95)

worker=Worker(input_size,
              output_size,
              architextures,
              transfer_rate,
              asyc_update_rate,
              gamma,
              learning_rate,
              Environment,
              sess,
              scope="Worker1",
              trainer=trainer)

T_MAX = 100
T = 0
sess.run(tf.global_variables_initializer())
worker.work(sess)

#state=worker.Environment.initialize_state()
#action=worker.Environment.random_action()
#reward, next_state, terminal = worker.Environment.slice.action(action)

#targets=worker.get_target(state, action, reward, next_state, terminal, sess)
#state = np.reshape(state, (1, len(state)))

#print("Before:")
#global_network.global_network.print_weights(sess)
#grads = sess.run(worker.gradients, feed_dict={worker.online_network.X_in: state, 
#                                              worker.targets: targets})
#print("Grads:")
#print(grads)
#feed_dict=dict(zip(worker.gradients_ph, grads))
#sess.run(worker.apply_grads, feed_dict=feed_dict)

#print("After:")
#global_network.global_network.print_weights(sess)

#worker.copy_global_weights(sess)
#worker.copy_weights()
#print("Done")