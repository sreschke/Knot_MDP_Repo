from SliceEnvironment import SliceEnv as SE
import numpy as np
import pandas as pd
import copy

class Start_States_Buffer(object):
    """A class that handles the start states used in the Knot MDP. Important members include the
    seed_frame and the explore_frame, which are both pandas dataframes. The column names of the 
    dataframes are stored in the class member columns. The seed_frame holds the data 
    corresponding to the knots we'd like to solve with the algorithm. The explore frame is
    initially identical to the seed_frame; as the algorithm explores states, they are added to 
    the explore_frame until the explore_frame reaches its capacity. After reaching its capacity,
    newer states displace states at the beginning of the buffer"""
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
        self.columns=["Braid", "Braid_Length", "Components", "Cursor", "Eulerchar", "Largest_Index"] #the columns of the dataframes
        self.seed_frame=self.construct_seed_frame()
        self.explore_frame=copy.copy(self.seed_frame)
        self.next_index=len(self.explore_frame)

    def construct_seed_frame(self):
        """Constructs the seed_frame using the seed_braids passed into the 
        constructor"""
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
        """Adds state data from slice to the explore_frame. We are currently using the .loc
        method to add rows, but there may be faster ways to accomplish this. Please see:
        https://stackoverflow.com/questions/41888080/python-efficient-way-to-add-rows-to-dataframe
        for a dataframe with potentially faster additions see racoon:
        https://github.com/rsheftel/raccoon"""

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
        #next_index=self.explore_frame.index.values.max()+1
        if len(self.explore_frame)<self.capacity:
            self.explore_frame.loc[self.next_index]=values
        else:
            self.explore_frame.iloc[self.next_index]=values
        self.next_index+=1
        if self.next_index==self.capacity:
            self.next_index=0
        #if len(self.explore_frame) > self.capacity:
        #    self.explore_frame=self.explore_frame.drop(self.explore_frame.index.values.min()) #drop first row
        assert len(self.explore_frame) <= self.capacity, "Start States Buffer has overfilled" 
        return

    def sample_state(self, largest_index=None, largest_length=None, frame="Explore"):
        """Samples a state from either the explore_frame or seed_frame using selections
        specified using the largest_index and largest_length parameters. For example, if
        largest_index=5 and largest_length=6, then the function will sample a state
        that has crossings no larger than 5 and a braid no longer than 6"""
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
        slice.inaction_penalty=self.move_penalty
        return slice