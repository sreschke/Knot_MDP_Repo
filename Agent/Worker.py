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
                gamma, #discount factor
                learning_rate,
                Environment,
                session,
                scope,
                trainer
                ):
        """This class is implemented to perform algorithm 1 from the paper: https://arxiv.org/pdf/1602.01783.pdf"""
        with tf.variable_scope(scope):
            self.online_network = DDQN(input_size=input_size,
                                       output_size=output_size,
                                       architextures=architextures,
                                       network_name="Online")
            self.target_network = DDQN(input_size=input_size,
                                       output_size=output_size,
                                       architextures=architextures,
                                       network_name="Target")
            self.Environment=Environment
            self.gamma=gamma
            self.epsilon=0.1
            self.learning_rate=tf.constant(learning_rate, dtype=tf.float32)
            self.transfer_rate=transfer_rate
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=0.95)
            self.session=session

            #place-holders
            self.targets = tf.placeholder(dtype=tf.float32, shape=(output_size,), name="Target")

            #operations
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.targets, self.online_network.forward_values_graph)))
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, local_vars)
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Global')
            self.apply_grads = trainer.apply_gradients(zip(self.gradients, global_vars))


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

    def work(self):
        pass
