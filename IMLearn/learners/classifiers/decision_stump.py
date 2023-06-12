from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_err = np.inf
        for j, feature in enumerate(X.T):
            for sign in [-1, 1]:
                thresh, err = self._find_threshold(feature, y, sign)
                if err < min_err:
                    min_err = err
                    self.threshold_ = thresh
                    self.sign_ = sign
                    self.j_ = j

        # err = np.inf
        # for j, sign in product(range(X.shape[1]), [-1, 1]):
        #     thresh, thresh_err = self._find_threshold(X[:, j], y, sign)
        #     if thresh_err < err:
        #         err = thresh_err
        #         self.threshold_ = thresh
        #         self.sign_ = sign
        #         self.j_ = j


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        if self.threshold_ is None:
            return np.zeros(X.shape[0])
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_,
                        -self.sign_)
        # return (np.array(X[:, self.j_] >= self.threshold_).astype(int) * 2 -
        #         1) * self.sign_


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        inds_sorted = np.argsort(values)
        vals_sorted, labels_sorted = values[inds_sorted], labels[inds_sorted]
        thresh_mat = np.concatenate(
            [[-np.inf], (vals_sorted[1:] + vals_sorted[:-1]) / 2, [-np.inf]])

        # abs_labels = np.abs(labels_sorted)
        # err = np.sum(abs_labels[np.sign(labels) == sign])
        err = np.abs(np.sum(labels_sorted[np.sign(labels_sorted) == sign]))
        err = np.append(err, err - np.cumsum(labels_sorted * sign))

        i_min_err = np.argmin(err)
        return thresh_mat[i_min_err], err[i_min_err]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return (y_pred != y).sum() / len(y)
