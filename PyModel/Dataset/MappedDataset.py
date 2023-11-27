from .BaseDataset import BaseDataset
import tensorflow as tf
from Transformer.params import Params, midi_test_params_v2
import numpy as np
import random
import pickle
import time

class ExampleData():
    def __init__(self,path,start_idx,end_idx):
        self.path = path
        self.start_idx = start_idx
        self.end_idx = end_idx

class MappedDataset(BaseDataset):
    pass