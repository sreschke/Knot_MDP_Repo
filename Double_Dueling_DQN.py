from SliceEnvironment import SliceEnv as SE
import tensorflow as tf
import numpy as np
import random
import copy
import time

from Dueling_DQN import Dueling_DQN as DDQN
from Uniform_Experience_Replay_Buffer import Uniform_Experience_Replay_Buffer as UERB


########################################################################################################################
#CLASS EXPLANATION: In this class, we have implemented the Double DQN algorithm with
#dueling architextures and experience replay, as described in the following papers:

#(1) Playing Atari with Deep Reinforcement Learning (2013)
#by Google DeepMind authored by Hado van Hasselt et al: https://arxiv.org/pdf/1509.06461.pdf
#Discusses the idea of a DQN and experience replay.

#(2) Deep Reinforcement Learning with Double Q-learning (2015)
#by Google DeepMind authored by Ziyu Wang et. al.: https://arxiv.org/pdf/1511.06581.pdf
#A detailed algorithm is given here: http://coach.nervanasys.com/algorithms/value_optimization/double_dqn/index.html
#Introduces the idea of using TWO networks, an online network and a target network. 
#This approach helps stabalize training by separating action selection from action 
#evaluation. The target network's weights are frozen while the online network is updated 
#at every time step. After every 1000 or so iterations, the online weights are copied to 
#the target weights.

#(3) Dueling Network Architectures for Deep Reinforcement Learning (2016)
#by Google DeepMind authored by Ziyu Wang et. al.: https://arxiv.org/pdf/1511.06581.pdf
#Introduces the Dueling Architexture (see figure 1 on page 1) which decouples estimates
#of the state value and advantage functions.

#(4) Prioritized Experience Replay (2016)
#by Google DeepMind authored by Tom Schaul et. al: https://arxiv.org/pdf/1511.05952.pdf
#Rather than uniformly sampling from the replay buffer, prioritized expereince replay 
#samples important (s, a, r, s', t) transitions more frequently which leads to more 
#efficient learning. We can anticipate a potential 2X speed-up.


#Neural Networks will allow the algorithm to generalize learning across similar
#states and thus allow us to tackle problems with larger state spaces while using
#less memory. The extra bells and whistles help to stabilize and speed-up training.
###################################################################################################
#Design
###################################################################################################
    #Important Members
    ###############################################################################################
    #A Double_Dueling_DQN (dddqn) has two Deuling_DQN objects: the online network and the target 
    #network. These are initialized in the __init__() method by passing in the parameters input_size, 
    #output_size, and architextures. The architextures parameter is a dictionary like 
    #the following:
    #architectures = {"Hidden": (128, 128, 64),
    #                "Value": (32, 1),
    #                "Advantage": (32, output_size)}
    #In this case, both networks will have an input layer of size input_size. The main section of the 
    #networks will have 3 hidden layers of sizes 128, 128, and 64. The value stream of the networks 
    #will have two layers with sizes 32 and 1 while the Advantage stream of the networks will have 
    #two layers with sizes 32 and output_size. The final layer of the value stream needs to have a size
    #of one.

    #The Double_Dueling_DQN also has an experience replay buffer called replay_buffer. This is an
    #object that must be passed into the constructor when declaring a dddqn object. The replay_buffer
    #must have the following members and methods:
        #members:
            #capacity
            #batch_size
        #methods:
            #add() -adds a (s, a, r, s', t) tuple to the buffer
            #get_batch() -returns batch_size (s, a, r, s', t) tuples
    
    #The Double_Dueling_DQN also has an Environment member which handles interactions between
    #the dddqn agent and the environment. This is an object that must be passed into the constructor 
    #when declaring a dddqn object. The Environment must have the following members and methods
        #members
            #num_actions
        #methods
            #initialize_state() -returns an encoded starting state
            #take_action() -takes a given action and returns the reward, next_state and terminal
            #random_action() -returns a random action
            #encode_state() -encodes the current state for input into a neural network
###################################################################################################

class Double_Dueling_DQN():
    def __init__(self,
                 input_size, #get this from len(encoded_state)
                 output_size, #the number of possible actions
                 architextures,
                 transfer_rate, #how often to copy the online weights to target weights
                 gamma, #discount factor
                 learning_rate,
                 Environment,
                 replay_buffer,
                 session
                 ):
        self.online_network = DDQN(input_size=input_size,
                                   output_size=output_size,
                                   architextures=architextures,
                                   network_name="Online")
        self.target_network = DDQN(input_size=input_size,
                                   output_size=output_size,
                                   architextures=architextures,
                                   network_name="Target")
        self.replay_buffer=replay_buffer
        self.Environment=Environment
        self.gamma=gamma
        self.epsilon=0.1
        self.learning_rate=tf.constant(learning_rate, dtype=tf.float32)
        self.transfer_rate=transfer_rate
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=0.95)
        self.session=session

        #Computation Graphs
        self.loss=self.get_loss()
        self.train_op=self.get_train_operation() #operation that updates the weights in the online network


    def initialize_replay_buffer(self, display=False, policy=None, euler_char_reset=-10):
        #Fills replay buffer to it's capacity by collecting (s, a, r, s', t) tuples. Actions are 
        #picked randomly until the buffer reaches its capacity.        
        state=self.Environment.initialize_state()
        if display:
            print("Filling replay buffer...")
        while len(self.replay_buffer.buffer) < self.replay_buffer.capacity:
            if policy is not None:
                action=self.Environment.action_from_policy(policy=policy)
            else:
                action=self.Environment.random_action()
            reward, next_state, terminal=self.Environment.take_action(action)
            self.replay_buffer.add((state, action, reward, next_state, terminal))
            if terminal or self.Environment.slice.eulerchar[1]<=euler_char_reset:
                state=self.Environment.initialize_state()
            else:
                state=next_state
            if display:
                print("Buffer Size: {} out of {}".format(len(self.replay_buffer.buffer), self.replay_buffer.capacity))
        if display:
            print("Replay buffer filled")
        return

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

    def train_step(self, current_states, actions, rewards, next_states, terminals, display_loss=False):
        targets=self.get_targets(current_states, actions, rewards, next_states, terminals, session=self.session) #Values
        #self.online_network.print_weights(self.session)
        self.session.run(self.train_op, feed_dict={self.online_network.X_in: current_states,
                                                   self.online_network.y_in: targets})
        #self.online_network.print_weights(self.session)
        if display_loss:
            print(self.session.run(self.loss, feed_dict={self.online_network.X_in: current_states,
                                                         self.online_network.y_in: targets}))
        return    
   
    def get_loss(self):
        #Calculates the MSE loss between the online network predictions and the targets.
        #Returns the computation graph for the loss.
        with tf.name_scope("Loss"):
            predictions=self.online_network.forward_values_graph #C graph
            loss = tf.losses.mean_squared_error(labels=self.online_network.y_in, predictions=predictions)
            train_op=self.optimizer.minimize(loss, global_step=self.global_step)
            return loss

    def get_train_operation(self):
        return self.optimizer.minimize(self.loss, global_step=self.global_step)

    def get_targets(self, current_states, actions, rewards, next_states, terminals, session=None):
        #implementation of the following algorithm: https://coach.nervanasys.com/algorithms/value_optimization/double_dqn/index.html
        #Calculates the target values given the (s, a, r, s', t) tuples in a mini_batch.
        #WARNING: the given algorithm does not take into account terminal transitions.
        #In these instances, the played_targets should be set to the reward.
        assert session is not None, "A tf.Session() must be passed into get_targets"
        with tf.name_scope("Targets"):
            #Using the NEXT STATES from the sampled batch, run the ONLINE network in order 
            #to find the Q maximizing action argmax_a_(Q(s', a) (action_indices)
            action_indices = session.run(self.online_network.forward_action_graph, feed_dict={self.online_network.X_in: next_states})
            #print(session.run(action_indices, feed_dict={self.online_network.X_in: next_states}))

            #For these actions, use the corresponding NEXT STATES and run the TARGET network 
            #to calculate the Q(s', argmax_a_(Q(s', a)) (selected_q_vals)
            all_t_q_vals = session.run(self.target_network.forward_values_graph, feed_dict={self.target_network.X_in: next_states})
            selected_t_q_vals = all_t_q_vals[range(len(all_t_q_vals)), action_indices]

            #In order to zero out the updates for the actions that were not played 
            #(resulting from zeroing the MSE loss), use the CURRENT STATES from the 
            #sampled batch, and run the ONLINE network to get the current Q values 
            #predictions. Set those values as the targets for the actions that were 
            #not actually played.
            targets = session.run(self.online_network.forward_values_graph, feed_dict={self.online_network.X_in: current_states})
            played_targets = np.array(rewards) + self.gamma*np.multiply(selected_t_q_vals, 1-np.array(terminals))
            targets[np.arange(len(targets)), actions] = played_targets
            return targets

    def copy_weights(self):
        #Copies the weights from the online network over to the target network. Rebuilds 
        #target_network's computation graph
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

    def print_weights_biases(self):
        #prints the weights and biases in both the online and target networks
        line_width=100
        print("="*line_width)
        print("Weights:")
        print("="*line_width)
        print("Online Network:")
        self.online_network.print_weights(self.session)
        print("="*line_width)
        print("Target Network:")
        print("="*line_width)
        self.target_network.print_weights(self.session)
        return
