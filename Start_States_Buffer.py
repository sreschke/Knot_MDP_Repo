from SliceEnvironment import SliceEnv as SE
import numpy as np
import pandas as pd

#Currently, we can't drop duplicate rows in the dataframe since lists and dictionaries 
#are unhashable. We might need to store everything in tuples if we want to drop duplicate
#rows.

class Start_States_Buffer(object):
    """A class that handles the start states used in the Knot MDP. The core datastructure is
    a pandas dataframe"""
    def __init__(self, seed_braids, max_braid_index, max_braid_length, capacity):
        self.seed_braids=seed_braids
        self.max_braid_length=max_braid_length
        self.max_braid_index=max_braid_index
        self.capacity=capacity
        self.columns=["Braid", "Braid_Length", "Components", "Cursor", "Eulerchar", "Largest_Index"]
        self.seed_frame=self.get_seed_frame()
        self.data_frame=pd.DataFrame(columns=self.columns)

    def get_seed_frame(self):
        seed_frame=pd.DataFrame(columns=self.columns) #initialize seed_frame
        for braid in self.seed_braids:
            slice=SE(braid_word=braid, 
                     max_braid_index=self.max_braid_index, 
                     max_braid_length=self.max_braid_length)
            values=[max(abs(slice.word)),
                    len(slice.word), 
                    slice.word,
                    slice.components,
                    slice.eulerchar,
                    slice.cursor]
            if len(seed_frame)==0:
                seed_frame=[dict(zip(self.columns, values))]
                seed_frame=pd.DataFrame(seed_frame)
            else: #append to to seed_frame
                temp_frame=[dict(zip(self.columns, values))]
                seed_frame=pd.concat([seed_frame, pd.DataFrame(temp_frame)])
        return seed_frame

    def add_state(self, slice):
        values=[max(abs(slice.word)),
                len(slice.word), 
                slice.word,
                slice.components,
                slice.eulerchar,
                slice.cursor]
        if len(self.data_frame)==0: 
            data_frame=[dict(zip(self.columns, values))]
            self.data_frame=pd.DataFrame(data_frame)
            return
        temp_frame=[dict(zip(self.columns, values))]
        self.data_frame=pd.concat([self.data_frame, pd.DataFrame(temp_frame)])
        if len(self.data_frame) > self.capacity:
            self.data_frame=self.data_frame.iloc[1:] #drop first row
        assert len(self.data_frame) <= self.capacity, "Start States Buffer has overfilled" 
        return

    def sample_state(self, largest_index, largest_length):
        pass

    def get_seed_slice(self):
        pass


seed_braids=[[1, 1, 1], [-1, -2, -3, 4]]
SSB=Start_States_Buffer(seed_braids, 7, 5, 100)
print(SSB.seed_frame)
