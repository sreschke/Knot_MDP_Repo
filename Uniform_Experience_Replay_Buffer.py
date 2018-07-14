from collections import deque
from Experience_Replay_Buffer import Experience_Replay_Buffer
import random

class Uniform_Experience_Replay_Buffer(Experience_Replay_Buffer):
    """Replay buffer implementation that uses uniform sampling. The essential data-structure is the 
    deque"""
    def __init__(self, capacity, batch_size):
        Experience_Replay_Buffer.__init__(self, capacity, batch_size)
        self.buffer=deque(maxlen=capacity)

    def add(self, to_add):
        #Adds to_add to buffer. If the buffer is full, to_add displaces the tuple at the end of the
        #deque
        self.buffer.appendleft(to_add)
        return

    def get_batch(self):
        assert len(self.buffer)!= 0, "Buffer is empty"
        return zip(*random.sample(self.buffer, self.batch_size)) #unpacks tuples