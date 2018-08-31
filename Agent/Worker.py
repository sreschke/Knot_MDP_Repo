from Dueling_DQN import Dueling_DQN as DDQN
import numpy as np
import random
import tensorflow as tf

class Worker(object):
    def __init__(self,
                input_size, #get this from len(encoded_state)
                output_size, #the number of possible actions
                architextures,
                transfer_rate, #how often to copy the online weights to target weights
                asyc_update_rate, #how often to update global network
                gamma, #discount factor
                learning_rate,
                Environment,
                session,
                scope,
                trainer
                ):
        """This class is implemented to perform algorithm 1 from the paper: https://arxiv.org/pdf/1602.01783.pdf
        Uses the github repository https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb as a reference"""
        
        #target network needs to be outside of variable scope so that we don't calculate the gradients with respect to its parameters
        self.target_network = DDQN(input_size=input_size,
                                   output_size=output_size,
                                   architextures=architextures,
                                   network_name="Target")

        with tf.variable_scope(scope):
            self.online_network = DDQN(input_size=input_size,
                                       output_size=output_size,
                                       architextures=architextures,
                                       network_name="Online")
            
            self.Environment=Environment
            self.gamma=gamma
            self.epsilon=0.1
            self.learning_rate=tf.constant(learning_rate, dtype=tf.float32)
            self.transfer_rate=transfer_rate
            self.asyc_update_rate=asyc_update_rate
            #self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.session=session

            #place-holders
            self.targets = tf.placeholder(dtype=tf.float32, shape=(output_size,), name="Target")

            #operations
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.targets, self.online_network.forward_values_graph)))
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            self.gradients_ph = []
            for tensor in self.gradients:
                self.gradients_ph.append(tf.placeholder(dtype=tf.float32, shape=tensor.shape))
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Global')
            self.apply_grads = trainer.apply_gradients(zip(self.gradients_ph, global_vars))
        self.update_local_ops = self.update_target_graph('Global', scope)

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
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

    def work(self, sess):
        self.copy_global_weights(sess)
        self.copy_weights()
        global T
        global T_MAX
        switch=True

        t=0
        state=self.Environment.initialize_state()
        
        while T < T_MAX:
            action=self.epsilon_greedy_action(state)
            reward, next_state, terminal = self.Environment.take_action(action)
            targets = self.get_target(state, action, reward, next_state, terminal, sess)
            state = np.reshape(state, (1, len(state)))

            #FIXME: figure out how to accumulate the gradients
            if switch:
                grads = sess.run(self.gradients, feed_dict={self.online_network.X_in: state,
                                                            self.targets: targets})
                switch=False
            else:
                to_add=sess.run(self.gradients, feed_dict={self.online_network.X_in: state,
                                                             self.targets: targets})
                #FIXME: there might be a faster way to do this next line. This is where we're accumulateing the grads
                grads=[np.array(list((map(np.add, x, y)))) for x, y in zip(grads, to_add)]

            state = next_state
            T+=1
            t+=1
            if T % self.transfer_rate == 0:
                self.copy_weights()
            if T % self.asyc_update_rate == 0:
                feed_dict=dict(zip(self.gradients_ph, grads))
                sess.run(self.apply_grads, feed_dict=feed_dict)
                switch=True
            


        print("Finished")

    def update_target_graph(self, from_scope,to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def copy_global_weights(self, session):
        """copies the global variables to the worker's online variables"""
        session.run(self.update_local_ops)
        return

    def copy_weights(self):
        #Copies the weights from the worker's online network over to the target 
        #network. Rebuilds target_network's computation graph
        #Copy hidden weights and biases
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