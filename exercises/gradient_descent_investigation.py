import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange]
        over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[
    1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the
    objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class,
        recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    vals, weights = [], []

    def wrapper(solver, w, val, grad, t, eta, delta):
        vals.append(val)
        weights.append(w)

    return wrapper, vals, weights


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    # Q 1,2
    conv1_traces, conv2_traces = [], []
    for eta in etas:
        # L1 module:
        cb1, vs1, ws1 = get_gd_state_recorder_callback()
        l1 = L1(init)
        best_w1 = GradientDescent(
            FixedLR(eta), out_type='best', callback=cb1).fit(l1)
        plot_descent_path(
            L1, np.stack(ws1),
            f"GD descent path: L1 module, fixed LR (eta={eta})"
            ).write_image(os.path.join(f"plt_q1_L1_{eta}.png"))

        conv1_traces.append(
            go.Scatter(x=[*range(len(vs1))], y=vs1,
                       mode=('markers+lines' if eta < .5 else 'markers'),
                       name=f"eta={eta}", line=dict(width=1),
                       marker=dict(size=3)))

        # L2 module:
        cb2, vs2, ws2 = get_gd_state_recorder_callback()
        l2 = L2(init)
        best_w2 = GradientDescent(
            FixedLR(eta), callback=cb2, out_type='best').fit(l2)

        plot_descent_path(
            L2, np.stack(ws2),
            f"GD descent path: L2 module, fixed LR (eta={eta})"
            ).write_image(os.path.join(f"plt_q1_L2_{eta}.png"))

        conv2_traces.append(
            go.Scatter(
                x=[*range(len(vs2))], y=vs2, mode='markers+lines',
                name=f"eta={eta}", line=dict(width=1), marker=dict(size=3)))

        # Q4
        min_loss1 = float(L1(best_w1).compute_output())
        min_loss2 = float(L2(best_w2).compute_output())
        print(f"using eta={eta}:\t L1 min loss is {min_loss1},\t"
              f"L2 min loss is {min_loss2}")

    # Q3
    go.Figure(conv1_traces).update_layout(
        title=f"GD: L1 convergence rate, fixed LR",
        xaxis=dict(title=f"iterations", showgrid=True),
        yaxis=dict(title=f"loss (L1 norm)", showgrid=True)
        ).write_image(os.path.join(f"plt_q3_L1.png"))
    go.Figure(conv2_traces).update_layout(
        title=f"GD: L2 convergence rate, fixed LR",
        xaxis=dict(title=f"iterations", showgrid=True),
        yaxis=dict(title=f"loss (squared L2 norm)", showgrid=True)
        ).write_image(os.path.join(f"plt_q3_L2.png"))


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate
    # raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    # raise NotImplementedError()

    # Plot descent path for gamma=0.95
    # raise NotImplementedError()
    pass


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train-
    and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples),
    n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples),
    n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease
    # data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models,
    # using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
