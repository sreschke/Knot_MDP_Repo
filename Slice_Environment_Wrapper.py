from SliceEnvironment import SliceEnv
import random
from Environment import Environment
from Start_States_Buffer import Start_States_Buffer as SSB
import copy

#An Environment Wrapper class needs to be defined for any given environment. Below is an implementation for
#the SliceEnvironment. The initialize_state(), take_action(), and random_action() functions need to 
#be implemented for each environment since these are called in the Double_Dueling_DQN methods.

class SliceEnvironmentWrapper(Environment):
    def __init__(self, max_braid_index=5, max_braid_length=7, inaction_penalty=0.05, start_states_buffer=None):
        Environment.__init__(self, num_actions=13)
        self.max_braid_index=max_braid_index
        self.max_braid_length=max_braid_length
        self.start_states_buffer=start_states_buffer
        self.slice=self.start_states_buffer.sample_state()

    def initialize_state(self):
        """When the algorithm reaches a terminal state, the state is initialized by calling
        this function. The start state gets pulled from the seed_frame with probability prob;
        otherwise, the state is pulled from the explore frame."""
        prob=0.5 #probability of sampling from seed_frame
        x=random.random()
        if x <= prob:
            self.slice=self.start_states_buffer.sample_state(frame="Seed")
        else:
            self.slice=self.start_states_buffer.sample_state(frame="Explore")
        return self.slice.encode_state()

    def take_action(self, action):
        reward, next_state, terminal=self.slice.action(action)
        if not terminal:
            self.start_states_buffer.add_state(copy.copy(self.slice))
        return reward, next_state, terminal

    def random_action(self):
        """Returns a random action. The actions in the sliceEnv MDP are categorized as "cursor moves",
        "shrinking moves", or "expanding moves". The action is pulled from shrinking_moves with
        probability probabilities[0]. The action is pulled from cursor_moves with probability
        probabilities[1]-probablities[0]. Otherwise, the action is pulled from expanding_moves"""
        shrinking_moves=[0, 8]
        cursor_moves = [1, 2, 3, 4]        
        expanding_moves=[5, 6, 7, 9, 10, 11, 12]
        x=random.random()
        probabilities=[0.3, 0.5]
        if x < probabilities[0]:
            return random.choice(shrinking_moves)
        elif x < probabilities[1]:
            return random.choice(cursor_moves)
        else:
            return random.choice(expanding_moves)