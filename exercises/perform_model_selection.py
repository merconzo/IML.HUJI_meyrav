from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model
    # f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    # raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[:n_samples], y[:n_samples]
    X_test, y_test = X[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_a = np.linspace(0.00001, 0.05, num=n_evaluations)
    lasso_a = np.linspace(0.0001, 1.75, num=n_evaluations)

    # train error, validation error
    ridge_t_err, ridge_v_err = np.zeros(n_evaluations), np.zeros(n_evaluations)
    lasso_t_err, lasso_v_err = np.zeros(n_evaluations), np.zeros(n_evaluations)
    for i in range(n_evaluations):
        cur_ridge_a, cur_lasso_a = ridge_a[i], lasso_a[i]
        ridge_mod = RidgeRegression(cur_ridge_a)
        lasso_mod = Lasso(cur_lasso_a, max_iter=5000)
        ridge_t_err[i], ridge_v_err[i] = cross_validate(
            ridge_mod, X_train, y_train, mean_square_error)
        lasso_t_err[i], lasso_v_err[i] = cross_validate(
            lasso_mod, X_train, y_train, mean_square_error)

    # RIDGE
    go.Figure(
        [go.Scatter(x=ridge_a, y=ridge_t_err,
                    mode='lines', marker=dict(color='purple'),
                    name='Train Error'),
         go.Scatter(x=ridge_a, y=ridge_v_err,
                    mode='lines', marker=dict(color='orange'),
                    name='Validation Error')
         ]).update_layout(
        title=f"Ridge: train & validation errors as a function of <br>the tested"
              f"regularization parameter value (n_evaluations={n_evaluations})",
        xaxis=dict(title=f"regularization parameter",
                   showgrid=True),
        yaxis=dict(title=f"error", showgrid=True)
        ).write_image(os.path.join(f"ridge.png"))

    # LASSO
    go.Figure(
        [go.Scatter(x=lasso_a, y=lasso_t_err,
                    mode='lines', marker=dict(color='cyan'),
                    name='Train Error'),
         go.Scatter(x=lasso_a, y=lasso_v_err,
                    mode='lines', marker=dict(color='rosybrown'),
                    name='Validation Error')
         ]).update_layout(
        title=f"Lasso: train & validation errors as a function of <br>the "
              f"tested "
              f"regularization parameter value (n_evaluations="
              f"{n_evaluations})",
        xaxis=dict(title=f"regularization parameter",
                   showgrid=True),
        yaxis=dict(title=f"error", showgrid=True)
        ).write_image(os.path.join(f"lasso.png"))

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_a = ridge_a[np.argmin(ridge_v_err)]
    lasso_best_a = lasso_a[np.argmin(lasso_v_err)]
    print("Ridge best regularization parameter: ", ridge_best_a)
    print("Lasso best regularization parameter: ", lasso_best_a)

    lin_mod = LinearRegression().fit(X_train, y_train)
    ridge_mod = RidgeRegression(lam=ridge_best_a).fit(X_train, y_train)
    lasso_mod = Lasso(alpha=lasso_best_a).fit(X_train, y_train)

    lin_loss = lin_mod.loss(X_test, y_test)
    ridge_loss = ridge_mod.loss(X_test, y_test)
    lasso_loss = mean_square_error(y_test, lasso_mod.predict(X_test))
    print("Ridge best validation error: ", ridge_loss)
    print("Lasso best validation error: ", lasso_loss)
    print("LinearRegression best validation error: ", lin_loss)


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
