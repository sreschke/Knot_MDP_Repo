from Experience_Replay_Buffer import Experience_Replay_Buffer
from prioritized_replay_memory import PrioritizedReplayMemory

class PERB_Wrapper(Experience_Replay_Buffer):
    def __init__(self, capacity, batch_size, epsilon, alpha, beta):
        assert batch_size<=capacity, "batch_size must be less than or equal to capacity"
        self.capacity=capacity
        self.batch_size=batch_size
        self.epsilon=epsilon
        self.PERB = PrioritizedReplayMemory(capacity, epsilon, alpha, beta)
        

    def add(self, to_add):
        """Adds to_add to the buffer with maximal priority.
        Returns the index to_add was assigned to"""
        index = self.PERB.add(to_add)
        return index

    def get_batch(self):
        weighted_samples = self.PERB.get_random_sample(self.batch_size)
        return zip(*weighted_samples)

    def get_size(self):
        return self.PERB.size()