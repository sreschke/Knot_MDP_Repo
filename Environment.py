from abc import ABCMeta, abstractmethod
#learn about abstract base classes in this tutorial: https://www.python-course.eu/python3_abstract_classes.php

class Environment(metaclass=ABCMeta):
    """Environment Abstract Base Class. Ensures that the methods initialize_state(), take_action()
    and random_action() are implemented in sub-classes"""
    def __init__(self, num_actions):
        self.num_actions=num_actions
        return

    @abstractmethod
    def initialize_state(self):
        state=None
        return state

    @abstractmethod
    def take_action(self, action):
        reward=None
        next_state=None
        terminal=None
        return reward, next_state, terminal

    @abstractmethod
    def random_action(self):
        return random.randint(0, self.num_actions-1)


