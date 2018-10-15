from Dueling_DQN import Dueling_DQN as DDQN
import tensorflow as tf
#import numpy as np
#import random


class Global_Network(object):
    """The global network"""
    def __init__(self, input_size, output_size, architextures, scope="Global"):
        with tf.variable_scope(scope):
            self.network=DDQN(input_size=input_size,
                                     output_size=output_size,
                                     architextures=architextures,
                                     network_name="Global")