import numpy as np
import heapq

#########################################################################################
#Look into this implementation:
##https://github.com/takoika/PrioritizedExperienceReplay
#########################################################################################

#########################################################################################
#Research Paper: https://arxiv.org/pdf/1511.05952.pdf
#########################################################################################
class Experience_Replay_Buffer(object):
    def __init__(self, capacity, batch_size):
        self.capacity=capacity
        self.batch_size=batch_size
        self.buffer = None

    def update(self):
        pass

    def get_batch(self):
        pass

