from abc import ABCMeta, abstractmethod
#learn about abstract base classes in this tutorial: https://www.python-course.eu/python3_abstract_classes.php

class Experience_Replay_Buffer(metaclass=ABCMeta):
    """Abstract Base Class for all experience replay objects. Ensures the methods add(), 
    get_batch(), and get_size() are implemented in sub-classes"""
    def __init__(self, capacity, batch_size):
        self.capacity=capacity
        self.batch_size=batch_size

    @abstractmethod
    def add(self, to_add):
        return None

    @abstractmethod
    def get_batch(self):
        sample=None
        return sample

    @abstractmethod
    def get_size(self):
        return None


