from Dueling_DQN import Dueling_DQN as DDQN
import numpy as np
import random
import tensorflow as tf


class Global_Network(object):
    """The global network"""
    def __init__(self, input_size, output_size, architextures, scope="Global"):
        with tf.variable_scope(scope):
            self.global_network=DDQN(input_size=input_size,
                                     output_size=output_size,
                                     architextures=architextures,
                                     network_name="Global")




