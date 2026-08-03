
from arcle.loaders import Loader
import numpy as np
import random


class BaseGridMaker(Loader):

    def __init__(self, **kwargs):

        # fix random seed
        rand_seed = kwargs['rand_seed']
        np.random.seed(rand_seed)
        random.seed(rand_seed)

        self._pathlist = self.get_path(**kwargs)
        self.data = self.convert_grid_to_uint8(self.parse(**kwargs))

    def get_path(self, **kwargs):
        return ['']

    def convert_grid_to_uint8(self, item):  # casting grid to uint8(for arcle 0.2.5)
        if isinstance(item, tuple):
            return tuple(self.convert_grid_to_uint8(elem) for elem in item)
        elif isinstance(item, dict):
            return {k: self.convert_grid_to_uint8(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self.convert_grid_to_uint8(elem) for elem in item]
        elif isinstance(item, np.ndarray):
            return np.array([self.convert_grid_to_uint8(elem) for elem in item])
        elif isinstance(item, np.integer):
            return np.uint8(item)
        elif isinstance(item, np.floating):
            return np.uint8(item)
        else:
            return item
