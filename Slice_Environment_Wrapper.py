from SliceEnvironment import SliceEnv
import random
from Environment import Environment
from Start_States_Buffer import Start_States_Buffer as SSB

#An Environment Wrapper class needs to be defined for any given environment. Below is an implementation for
#the SliceEnvironment. The initialize_state(), take_action(), and random_action() functions need to 
#be implemented for each environment since these are called in the Double_Dueling_DQN methods.

class SliceEnvironmentWrapper(Environment):
    def __init__(self, max_braid_index=5, max_braid_length=7, inaction_penalty=0.05, start_states_buffer=None):
        Environment.__init__(self, num_actions=13)
        self.max_braid_index=max_braid_index
        self.max_braid_length=max_braid_length
        self.start_states_buffer=start_states_buffer
        self.start_state=self.initialize_state()
        self.slice=SliceEnv(braid_word=self.start_word, 
                            max_braid_index=self.max_braid_index,
                            max_braid_length=self.max_braid_length,
                            inaction_penalty=inaction_penalty)
    def initialize_state(self):
        braid=random.choice(self.starting_braids)
        self.slice=SliceEnv(braid_word=braid, max_braid_index=self.max_braid_index, max_braid_length=self.max_braid_length)
        return self.slice.encode_state()
    def take_action(self, action):
        reward, next_state, terminal=self.slice.action(action)
        return reward, next_state, terminal
    def random_action(self):
        #May need to adjust this so that shrinking and lengthing actions are picked with equal 
        #probability
        cursor_moves = [1, 2, 3, 4]
        shrinking_moves=[0, 8]
        expanding_moves=[5, 6, 7, 9, 10, 11, 12]
        x=random.random()
        probabilities=[0.3, 0.5]
        if x < probabilities[0]:
            return random.choice(shrinking_moves)
        elif x < probabilities[1]:
            return random.choice(cursor_moves)
        else:
            return random.choice(expanding_moves)