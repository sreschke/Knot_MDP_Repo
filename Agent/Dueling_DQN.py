import tensorflow as tf
import copy

class Dueling_DQN(object):
    """A neural network class deuling archtexture: standard body as well
    as value and advantage streams"""
    def __init__(self, input_size, output_size, architextures, network_name):
        assert (architextures["Value"][-1]==1), "Last element of value_architextures must be 1!"
        assert (architextures["Advantage"][-1]==output_size), "Last element of advantage_architextures must equal output_size!"
        self.input_size=input_size
        self.output_size=output_size
        self.network_name=network_name

        #architextures
        self.architextures=architextures #a dictionary with "Hidden", "Value", and "Advantage" keys mapping to tuples

        #Weights and Biases
        self.hidden_weights, self.hidden_biases=self.construct_weights_biases(self.input_size, self.architextures["Hidden"], section_name="Hidden")
        self.value_weights, self.value_biases=self.construct_weights_biases(self.architextures["Hidden"][-1], self.architextures["Value"], section_name="Value")
        self.advantage_weights, self.advantage_biases=self.construct_weights_biases(self.architextures["Hidden"][-1], self.architextures["Advantage"], section_name="Advantage")
        
        #place-holders
        self.X_in=tf.placeholder(dtype=tf.float32, shape=(None, input_size), name="Input")
        self.y_in=tf.placeholder(dtype=tf.float32, shape=(None, output_size), name="Targets")

        #Computation_Graphs
        self.forward_values_graph, self.forward_action_graph=self.forward()

    def construct_weights_biases(self, input_size, architextures, section_name):
        #Declares variables for the weights and biases of layers in the network 
        #and returns them in lists.

        #The name scopes help organize the computation graphs in Tensorboard
        with tf.name_scope(section_name):
            weights = []
            biases = []
            #Declare hidden weights and biases for first layer
            weights.append(tf.get_variable(name=self.network_name+"_"+section_name+"_Weights0",
                    shape=[input_size, architextures[0]],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32))
            biases.append(tf.Variable(initial_value=tf.zeros([architextures[0]]),
                    name=self.network_name+"_"+section_name+"Bias0",
                    dtype=tf.float32))

            #Declare hidden weights and biases for hidden layers
            for i in range(1, len(architextures)):
                #get names for layer i
                matrix_name=self.network_name+"_"+section_name+"Weights{}".format(i)
                bias_name=self.network_name+"_"+section_name+"Bias{}".format(i)

                #declare hidden weights and bias for layer i
                weights.append(tf.get_variable(name=matrix_name,
                        shape=[architextures[i-1], architextures[i]],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32))
                biases.append(tf.Variable(initial_value=tf.zeros([architextures[i]]),
                    name=bias_name,
                    dtype=tf.float32))

            return weights, biases

    def forward(self):
        """Builds the computation graphs for the q_values and argmax of q_values."""
        with tf.name_scope("Forward"):
            #construct computation graph
            #hidden layers
            with tf.name_scope("Hidden"):
                activation = self.X_in
                for i in range(0, len(self.hidden_weights)):
                    preActivation = tf.add(tf.matmul(activation, self.hidden_weights[i]), self.hidden_biases[i])
                    activation = tf.nn.relu(preActivation)
                hidden_activation = activation

            #value stream
            with tf.name_scope("Value"):
                value = hidden_activation
                for i in range(0, len(self.value_weights)-1):
                    preV = tf.add(tf.matmul(value, self.value_weights[i]), self.value_biases[i])
                    value = tf.nn.relu(preV)
                state_value = tf.add(tf.matmul(value, self.value_weights[-1]), self.value_biases[-1])
    
            #advantage stream
            with tf.name_scope("Advantage"):
                advantage = hidden_activation
                for i in range(0, len(self.advantage_weights)-1):
                    preA = tf.add(tf.matmul(advantage, self.advantage_weights[i]), self.advantage_biases[i])
                    advantage = tf.nn.relu(preA)
                state_advantages = tf.add(tf.matmul(advantage, self.advantage_weights[-1]), self.advantage_biases[-1])
        
            #Combine value and adantage streams to get Q_Values and optimal action using 
            #equation (9) from Ziyu Wang et. al. paper
            with tf.name_scope("Q_Values"):
                q_values = tf.add(state_value, tf.add(state_advantages, tf.div(-tf.reduce_sum(state_advantages), tf.cast(tf.size(state_advantages), tf.float32))))
                opt_action = tf.argmax(q_values, axis=1, output_type=tf.int32)
            
            #return the computation graphs
            return q_values, opt_action

    def print_weights(self, session):
        """Print the weights and biases in main body, advantage stream,
        and value stream"""
        line_length=100
        print("="*line_length)
        print("Hidden Layers")
        print("="*line_length)
        for i in range(len(self.hidden_weights)):
            print("Hidden Weights Layer {}:".format(i))
            print("{}".format(session.run(self.hidden_weights[i])))
            print()
            print("Hidden Bias Layer {}:".format(i))
            print("{}".format(session.run(self.hidden_biases[i])))
            print()

        print("="*line_length)
        print("Advantage Layers")
        print("="*line_length)
        for j in range(len(self.advantage_weights)):
            print("Advantage Weights Layer {}:".format(i))
            print("{}".format(session.run(self.advantage_weights[j])))
            print()
            print("Advantage Bias Layer {}:".format(i))
            print("{}".format(session.run(self.advantage_biases[j])))
            print()

        print("="*line_length)
        print("Value Layers")
        print("="*line_length)
        for k in range(len(self.value_weights)):
            print("Value Weights Layer {}:".format(i))
            print("{}".format(session.run(self.value_weights[k])))
            print()
            print("Value Bias Layer {}:".format(i))
            print("{}".format(session.run(self.value_biases[k])))
            print()
        return