
import sklearn.decomposition
# Setup which base classes to consider.
sklearn_pca = [sklearn.decomposition.PCA, sklearn.decomposition.TruncatedSVD,
               sklearn.decomposition.KernelPCA,
               sklearn.decomposition.IncrementalPCA,
               sklearn.decomposition.SparsePCA,
               sklearn.decomposition.MiniBatchSparsePCA]
default_sklearn_pca = [sklearn.decomposition.PCA, sklearn.decomposition.TruncatedSVD]
from .api import PCA

