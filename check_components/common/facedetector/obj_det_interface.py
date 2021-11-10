from abc import ABCMeta, abstractmethod


class ObjectDetector(metaclass=ABCMeta):

    def __init__(self, cfg):
        self._cfg = cfg
        self._detector = None

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def detect(self):
        pass

    @abstractmethod
    def plot_boxes(self):
        pass