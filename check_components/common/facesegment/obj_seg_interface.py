from abc import ABCMeta, abstractmethod


class ObjectSegmentation(metaclass=ABCMeta):

    def __init__(self, cfg):
        self._cfg = cfg
        self._segmentor = None

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def segment(self):
        pass

    # @abstractmethod
    # def plot_boxes(self):
    #     pass
