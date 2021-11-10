from abc import ABCMeta, abstractmethod


class LandmarkDetector(metaclass = ABCMeta):

    def __init__(self,cfg):
        self._cfg = cfg
        self._detector = None
    
    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def return_coords(self):
        pass

    @abstractmethod
    def show_landmarks(self):
        pass 

    @abstractmethod
    def get_points(self):
        pass 