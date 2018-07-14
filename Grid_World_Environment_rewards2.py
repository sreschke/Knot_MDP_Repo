from Environment import Environment
import numpy as np
import copy
import random

class GridWorldEnvironment_rewards2(Environment):
    """Small Environment to test Double_Dueling_DQN"""
    def __init__(self, num_actions, game_map, rewards_map, penalty=0.05):
        assert game_map.shape == rewards_map.shape, "game_map and rewards must have the same shape"
        Environment.__init__(self, num_actions)
        self.actions=["U", "D", "L", "R"]
        self.action_map={0: "U", 1: "D", 2: "L", 3: "R"}
        self.game_map=game_map
        self.num_rows=len(game_map)
        self.num_columns=len(game_map[0])
        self.start_states = self.get_start_states()
        self.state=random.choice(self.start_states)
        self.rewards_map=rewards_map
        self.penalty=penalty
        self.big_penalty=2

    def get_start_states(self):
        start_states=[]
        for row in range(self.num_rows):
            for column in range(self.num_columns):
                if self.game_map[row][column]==1:
                    start_states.append([row, column])
        return start_states

    def initialize_state(self):
        self.state=copy.copy(random.choice(self.start_states))
        return self.encode_state()
    
    ######################################################################################
    #take_action() using original rewards definition
    ######################################################################################
    def take_action(self, action):
        #Takes the given action (changes internal state) and returns the corresponding
        #reward, next_state, and terminal 
        old_state=copy.copy(self.state)
        if action in [0, "U"]:
            self.state[0] = self.state[0]-1
        elif action in [1, "D"]:
            self.state[0] = self.state[0]+1
        elif action in [2, "L"]:
            self.state[1] = self.state[1]-1
        elif action in [3, "R"]:
            self.state[1] = self.state[1]+1
        else:
            assert True==False, "Invalid action passed to take_action()"
        if self.state[0] < 0 \
            or self.state[0] > self.num_rows-1 \
            or self.state[1] < 0 \
            or self.state[1] > self.num_columns-1:
            self.state=old_state
        if self.game_map[self.state[0]][self.state[1]]==0: #if we run into an obstacle
            self.state=old_state
        if self.state==old_state: #if we ran into a wall
            reward=-self.big_penalty
        else:
            reward=self.rewards_map[self.state[0]][self.state[1]]-self.rewards_map[old_state[0]][old_state[1]]-self.penalty
        if self.game_map[self.state[0]][self.state[1]]==-1:
            terminal=1
        else:
            terminal=0
        return reward, self.encode_state(), terminal

    def random_action(self):
        return random.choice(range(self.num_actions))

    def encode_state(self, one=1, zero=0, display=False):
        row_encoding=np.ones(self.num_rows)*zero
        row_encoding[self.state[0]]=one
        column_encoding=np.ones(self.num_columns)*zero
        column_encoding[self.state[1]]=one
        full_encoding=np.concatenate([row_encoding, column_encoding])
        if display:
            print("Row encoding: {}".format(row_encoding))
            print("Column encoding: {}".format(column_encoding))
            print("Full encoding: {}".format(full_encoding))
        return full_encoding
    
    def set_state(self, new_state):
        assert(new_state[0]>=0 and new_state[0]<self.num_rows), "Row index must satisfy 0<row_index<num_rows"
        assert(new_state[1]>=0 and new_state[1]<self.num_columns), "Column index must satisfy 0<column_index<num_columns"
        self.state=new_state

    def print_map(self, position_marker="O", obstacle_marker="X"):
        for row in range(self.num_rows):
            print("|", end="")
            for col in range(self.num_columns):
                reward = self.rewards_map[row][col]
                if self.game_map[row][col]==0:
                    print("{0:2}|".format(obstacle_marker), end="")
                elif self.state == [row, col]:
                    print("{0:2}|".format(position_marker), end="")
                elif reward != 0:
                    print("{0:2}|".format(reward), end="")
                else:
                    print("{0:2}|".format(""), end="")
            print()
        return
        
    def action_from_policy(self, policy):
        return policy[self.state[0]][self.state[1]]

