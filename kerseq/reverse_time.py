# import theano.tensor as T
from keras.layers.core import MaskedLayer

class ReverseTime(MaskedLayer):
    '''
        Reverse order of samples in time-series.
        Assumes input is a 3D array with dims (n_samples, n_time, n_features)
    '''
    def __init__(self):
        super(ReverseTime, self).__init__()

    def get_output(self, train=False):
        X = self.get_input(train)
        return X[:, ::-1]

    def get_output_mask(self, train=False):
        input_mask = self.get_input_mask(train)
        return input_mask[:, ::-1]
