from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    indices = np.arange(X.shape[0])
    sub_locs = np.array_split(indices, cv)

    train_score, valid_score = 0.0, 0.0
    for locs in sub_locs:
        i_train = np.setdiff1d(indices, locs)
        cur_train_X, cur_train_y = X[i_train], y[i_train]
        fitted_copy = deepcopy(estimator).fit(cur_train_X, cur_train_y)
        cur_pred_y = fitted_copy.predict(cur_train_X)
        train_score += scoring(cur_train_y, cur_pred_y)

        cur_X, cur_y = X[locs], y[locs]
        cur_pred_y = fitted_copy.predict(cur_X)
        valid_score += scoring(cur_y, cur_pred_y)

    return (train_score / cv), (valid_score / cv)

