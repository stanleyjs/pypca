import numpy as np
import sklearn.decomposition as skld
from sklearn.base import BaseEstimator

def get_pca(base_class):
    class PCA(*base_class):
        def __init__(self, base_class=None, verbose=True, **kwargs):
            self.base_class = base_class
            self.verbose = verbose
    return PCA
