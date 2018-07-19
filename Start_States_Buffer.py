from SliceEnvironment import SliceEnv as SE
import numpy as np
import pandas as pd
import copy

class Start_States_Buffer(object):
    """A class that handles the start states used in the Knot MDP. The core datastructure is
    a pandas dataframe."""
    def __init__(self, seed_braids, max_braid_index, max_braid_length, capacity):
        for braid in seed_braids:
            assert max(abs(np.array(braid)))<=max_braid_index-1, "Cannot initialize braid {} with max_braid_index {}".format(braid, max_braid_index)
        self.seed_braids=seed_braids
        self.max_braid_length=max_braid_length
        self.max_braid_index=max_braid_index
        self.capacity=capacity
        self.columns=["Braid", "Braid_Length", "Components", "Cursor", "Eulerchar", "Largest_Index"]
        self.seed_frame=self.get_seed_frame()
        self.explore_frame=copy.copy(self.seed_frame)
        #self.explore_frame=pd.DataFrame(columns=self.columns)

    def get_seed_frame(self):
        seed_frame=pd.DataFrame(columns=self.columns) #initialize seed_frame
        for braid in self.seed_braids:
            slice=SE(braid_word=braid, 
                     max_braid_index=self.max_braid_index, 
                     max_braid_length=self.max_braid_length)
            values=[slice.word, #Braid
                    len(slice.word), #Braid_Length
                    slice.components, #Components
                    slice.cursor, #Cursor
                    slice.eulerchar, #Eulerchar
                    max(abs(slice.word))] #Largest_Index
            if len(seed_frame)==0:
                seed_frame=[dict(zip(self.columns, values))]
                seed_frame=pd.DataFrame(seed_frame)
            else: #append to to seed_frame
                temp_frame=[dict(zip(self.columns, values))]
                seed_frame=pd.concat([seed_frame, pd.DataFrame(temp_frame)])
        return seed_frame

    def add_state(self, slice):
        #adds state data from slice to the explore_frame
        values=[copy.copy(slice.word), #Braid
               copy.copy(len(slice.word)), #Braid_Length
               copy.copy(slice.components), #Components
               copy.copy(slice.cursor), #Cursor
               copy.copy(slice.eulerchar), #Eulerchar
               copy.copy(max(abs(slice.word)))] #Largest_Index
        if len(self.explore_frame)==0: 
            explore_frame=[dict(zip(self.columns, values))]
            self.explore_frame=pd.DataFrame(explore_frame)
            return
        #temp_frame=[dict(zip(self.columns, values))]
        #self.explore_frame=pd.concat([self.explore_frame, pd.DataFrame(temp_frame)])
        next_index=self.explore_frame.index.values.max()+1
        self.explore_frame.loc[next_index]=values
        if len(self.explore_frame) > self.capacity:
            self.explore_frame=self.explore_frame.drop(self.explore_frame.index.values.min()) #drop first row
        assert len(self.explore_frame) <= self.capacity, "Start States Buffer has overfilled" 
        return

    def sample_state(self, largest_index=None, largest_length=None, frame="Explore"):
        """Samples a state from either the explore_frame or seed_frame using selections
        specified using the largest_index and largest_length parameters"""
        assert frame in ["Explore", "Seed"], "frame flag must be either \"Explore\" or \"Seed\""
        #get appropriate data_frame
        if frame=="Explore":
            df=self.explore_frame
        else:
            df=self.seed_frame
        assert len(df)>0, "Can't sample from an empty dataframe. Use the add_state() function to add to the explore_frame."
        
        #select from data_frame
        if largest_index is None:
            if largest_length is None:
                pass
            else:
                df=df.loc[(df["Braid_Length"] <= largest_length)]
        else:
            if largest_length is None:
                df=df.loc[(df["Largest_Index"] <= largest_index)]
            else:
                df=df.loc[(df['Largest_Index'] <= largest_index)
                          & (df["Braid_Length"] <= largest_length)]
        assert len(df)>0, "len(df)=0 with current selections" 
        df=df.sample()
        slice=SE(braid_word=df["Braid"].iloc[0],
                 max_braid_index=self.max_braid_index,
                 max_braid_length=self.max_braid_length)
        slice.components=copy.copy(df["Components"].iloc[0])
        slice.eulerchar=copy.copy(df["Eulerchar"].iloc[0])
        slice.cursor=copy.copy(df["Cursor"].iloc[0])
        return slice
