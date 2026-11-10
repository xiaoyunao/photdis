"""One prespecified class-partition candidate with natural leaf-member mass."""
import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .baseline_v1_2 import balanced_weights


def class_partition_candidate(control, mags, luminosity, labels):
    """Keep identities, histogram grid and RF complexity fixed; change split loss.

    Balanced class weights affect partition learning only. Every original PSF
    training member contributes unit natural weight to the distance histogram.
    Neither evaluation labels nor true class enter prediction.
    """
    if control.forest.bootstrap:
        raise ValueError('This experiment fixes bootstrap=False for both arms')
    params = control.forest.get_params()
    names = ['n_estimators', 'min_samples_leaf', 'max_depth', 'max_features',
             'bootstrap', 'random_state', 'n_jobs']
    candidate = copy.copy(control)
    candidate.forest = RandomForestClassifier(**{k: params[k] for k in names})
    x = control.feature_matrix(mags)
    candidate.forest.fit(x, labels, sample_weight=balanced_weights(labels))
    candidate.tables, candidate.types, candidate.maps = [], [], []
    nb = len(control.centers)
    bins = np.searchsorted(control.edges, luminosity, side='right')-1
    for tree in candidate.forest.estimators_:
        leaf = tree.apply(x)
        unique, inverse = np.unique(leaf, return_inverse=True)
        mapping = np.zeros(tree.tree_.node_count, dtype='i4')
        mapping[unique] = np.arange(len(unique))
        histogram = np.bincount(inverse*nb+bins, minlength=len(unique)*nb).reshape(-1, nb).astype('f4')
        histogram /= histogram.sum(axis=1)[:, None]
        classes = np.bincount(inverse*4+labels, minlength=len(unique)*4).reshape(-1, 4).astype('f4')
        classes /= classes.sum(axis=1)[:, None]
        candidate.maps.append(mapping)
        candidate.tables.append(histogram)
        candidate.types.append(classes)
    return candidate
