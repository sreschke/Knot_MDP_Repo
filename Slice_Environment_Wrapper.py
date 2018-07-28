from SliceEnvironment import SliceEnv
import random
from Environment import Environment
from Start_States_Buffer import Start_States_Buffer as SSB
import copy

#An Environment Wrapper class needs to be defined for any given environment. Below is an implementation for
#the SliceEnvironment. The initialize_state(), take_action(), and random_action() functions need to 
#be implemented for each environment since these are called in the Double_Dueling_DQN methods.

class Slice_Environment_Wrapper(Environment):
    def __init__(self,
                 max_braid_index=5,
                 max_braid_length=7,
                 inaction_penalty=0.05,
                 start_states_buffer=None,
                 action_probabilities=[0.3, 0.5],
                 seed_prob=0.5,
                 uniform=False):
        assert type(start_states_buffer) is not None, "A start_states_buffer must be passed into the SliceEnvironmentWrapper constructor"
        assert len(action_probabilities)==2, "Length of action_probabilities must be 2"
        Environment.__init__(self, num_actions=13)
        self.max_braid_index=max_braid_index
        self.max_braid_length=max_braid_length
        self.start_states_buffer=start_states_buffer
        self.slice=self.start_states_buffer.sample_state()
        self.action_probabilities=action_probabilities
        self.seed_prob=seed_prob
        self.uniform=uniform

    def initialize_state(self):
        """When the algorithm reaches a terminal state, the state is initialized by calling
        this function. The start state gets pulled from the seed_frame with probability 
        seed_prob; otherwise, the state is pulled from the explore frame."""
        x=random.random()
        if x <= self.seed_prob:
            self.slice=self.start_states_buffer.sample_state(frame="Seed")
        else:
            self.slice=self.start_states_buffer.sample_state(frame="Explore")
        return self.slice.encode_state()

    def take_action(self, action):
        """Takes action and adds the resulting state to the start states buffer."""
        reward, next_state, terminal=self.slice.action(action)
        if not terminal:
            self.start_states_buffer.add_state(copy.copy(self.slice))
        return reward, next_state, terminal

    def random_action(self):
        """Returns a random action. Actions are sampled uniformly if self.uniform=True. Otherwise,
        actions are pulled from distribution defined by self.action_probabilities: the actions 
        in the sliceEnv MDP are categorized as "cursor moves", shrinking moves", or "expanding 
        moves". The action is pulled from shrinking_moves with probability 
        action_probabilities[0]. The action is pulled from cursor_moves with probability
        action_probabilities[1]-action_probablities[0]. Otherwise, the action is pulled from 
        expanding_moves."""
        if self.uniform:
            return random.choice(range(14))
        else:
            for prob in self.action_probabilities:
                assert prob <= 1 and prob >= 0, "Check action_probabilities"
            assert self.action_probabilities[0]<=self.action_probabilities[1], "Check action_probabilites"
            shrinking_moves=[0, 8] #remove crossing and remove r2
            cursor_moves = [1, 2, 3, 4] #move up, down, left, right
            expanding_moves=[5, 6, 7, 9, 10, 11, 12] #cut, add positve/negative r2, r3, far comm, add/remove crossing
            x=random.random()
            if x < self.action_probabilities[0]:
                return random.choice(shrinking_moves)
            elif x < self.action_probabilities[1]:
                return random.choice(cursor_moves)
            else:
                return random.choice(expanding_moves)