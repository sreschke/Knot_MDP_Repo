from SliceEnvironment import SliceEnv as SE
from collections import deque
import numpy as np
import copy
import random

class Start_States_Buffer(object):
    """A class that handles the start states used in the Knot MDP. As the algorithm explores 
    states, they are added to the explore_queue until the explore_frame reaches its capacity.
    After reaching its capacity, new states displace states at the end of the buffer.
    
    This newer implementation uses queues rather than pandas dataframes and performs additions
    roughly 150 times faster"""
    def __init__(self, seed_braids, max_braid_index, max_braid_length, capacity, move_penalty):
        #Ensure seed_braids are compatable with max_braid_index and max_braid_length
        for braid in seed_braids:
            assert max(abs(np.array(braid)))<=max_braid_index-1, "Cannot initialize braid {} with max_braid_index {}".format(braid, max_braid_index)
            assert len(braid)<=max_braid_length, "Cannot initialize braid {} with max_braid_length {}".format(braid, max_braid_length)
        self.seed_braids=seed_braids #a list of braids we'd like the algorithm to solve
        self.max_braid_length=max_braid_length
        self.max_braid_index=max_braid_index
        self.capacity=capacity #the capacity of the explore_frame
        self.move_penalty=move_penalty
        self.seed_queue=self.construct_seed_queue()
        self.explore_queue=copy.copy(self.seed_queue)

    def construct_seed_queue(self):
        temp_queue=deque(maxlen=len(self.seed_braids))
        for braid in self.seed_braids:
            slice=SE(braid_word=braid,
                     max_braid_index=self.max_braid_index,
                     max_braid_length=self.max_braid_length)
            temp_queue.appendleft(slice.encode_state())
        return temp_queue

    def add_state(self, slice):
        """Adds state data from slice to the explore_queue."""
        self.explore_queue.appendleft(slice.encode_state())        
        assert len(self.explore_queue) <= self.capacity, "explore_que has overfilled" 
        return

    def inverse_encode_state(self, encoded_state, zero=0, one=1):
        """Takes an encoded state and returns braid, components, eulerchar, and cursor info to
        construct a sliceEnv obj."""

        #cast to np.array
        encoded_state=np.array(encoded_state)
        #get braid word
        crossing_encoding_length=2*(self.max_braid_index)-1
        braid_encoding_length=crossing_encoding_length*self.max_braid_length
        braid_encoding=encoded_state[:braid_encoding_length]
        braid=[]
        #print("braid_encoding_length: {}".format(braid_encoding_length))
        #print("crossing_encoding_length: {}".format(crossing_encoding_length))
        #print("braid_encoding_length/crossing_encoding_length: {}".format(braid_encoding_length/crossing_encoding_length))
        #print("braid_encoding: {}".format(braid_encoding))
        for crossing_encoding in np.split(ary=braid_encoding,
                                          indices_or_sections=int(braid_encoding_length/crossing_encoding_length)):
            braid.append(np.where(crossing_encoding==one)[0][0])
        zero_index=self.max_braid_index-1
        braid[:]=[x-zero_index for x in braid]
        braid[:]=[x for x in braid if x!=0]
        braid=np.array(braid)

        #get components
        components_encoding_length=self.max_braid_index*(self.max_braid_index+1)
        next_index=braid_encoding_length
        components_encoding=encoded_state[next_index:next_index+components_encoding_length]
        components=[]
        for component_encoding in np.split(components_encoding, self.max_braid_index):
            components.append(np.where(component_encoding==one)[0][0])
        components[:]=[x+1 for x in components]
        components=np.array(components)

        #get eulerchar
        eulerchar_encoding_length=len(components)
        next_index+=components_encoding_length
        euler_values=encoded_state[next_index:next_index+eulerchar_encoding_length]
        eulerchar=dict(zip(components, euler_values))
        
        #get row
        row_encoding_length=self.max_braid_length+1
        next_index+=eulerchar_encoding_length
        row_encoding=encoded_state[next_index:next_index+row_encoding_length]
        row=np.where(row_encoding==one)[0][0]

        #get column
        column_encoding_length=self.max_braid_index-1
        next_index+=row_encoding_length
        column_encoding=encoded_state[next_index:next_index+column_encoding_length]
        column=np.where(column_encoding==one)[0][0]+1

        #create cursor
        cursor=np.array([row, column])

        return braid, components, eulerchar, cursor

    def sample_state(self, queue="Explore"):
        """Samples a state from the explore_queue if queue="Explore".
        Samples a state from seed_queue if queue="Seed"."""
        
        assert queue in ["Explore", "Seed"], "queue flag must be either \"Explore\" or \"Seed\""
        #get appropriate data_frame
        if queue=="Explore":
            encoded_state=random.sample(population=list(self.explore_queue), k=1)[0]
        else:
            encoded_state=random.sample(population=list(self.seed_queue), k=1)[0]
        #print("Encoded state in sample_state: {}".format(encoded_state))
        braid, components, eulerchar, cursor = self.inverse_encode_state(encoded_state=encoded_state)
        slice=SE(braid_word=braid,
                    max_braid_index=self.max_braid_index,
                    max_braid_length=self.max_braid_length,
                    inaction_penalty=self.move_penalty)
        slice.components=components
        slice.eulerchar=eulerchar
        slice.cursor=cursor
        return slice




