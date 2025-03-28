import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def generate_data(n: int, noise_ratio: float = 0) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape 
    (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(
        train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_learner = AdaBoost(wl=DecisionStump, iterations=n_learners).fit(
        train_X, train_y)
    iters = [*range(1, n_learners + 1)]
    train_losses = [ada_learner.partial_loss(train_X, train_y, t) for t in
                    iters]
    test_losses = [ada_learner.partial_loss(test_X, test_y, t) for t in iters]

    go.Figure(
        [go.Scatter(x=iters, y=train_losses,
                    mode='lines', marker=dict(color='purple'),
                    name='Train Loss'),
         go.Scatter(x=iters, y=test_losses,
                    mode='lines', marker=dict(color='orange'),
                    name='Test Loss')
         ]).update_layout(
        title=f"Change in Adaboost's Loss over Number of Classifiers ("
              f"noise={noise})",
        xaxis=dict(title=f"Iteration (number of fitted learners)",
                   showgrid=True),
        yaxis=dict(title=f"Classification error", showgrid=True)
        ).write_image(os.path.join(f"plot_ada_{noise}.png"))

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    locs = [(1, 1), (1, 2), (2, 1), (2, 2)]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    fig = make_subplots(2, 2, subplot_titles=[f"{t} classifiers" for t in T],
                        vertical_spacing=0.1)
    for loc, t in zip(locs, T):
        r, c = loc
        surface = decision_surface(
            predict=(lambda X: ada_learner.partial_predict(X, t)),
            xrange=lims[0], yrange=lims[1],  showscale=False, density=50)
        fig.add_traces(
            [surface,
             go.Scatter(
                 x=test_X[:, 0], y=test_X[:, 1],
                 mode="markers",
                 marker=dict(
                     color=test_y,
                     symbol=np.where(test_y == 1, class_symbols[0],
                                     class_symbols[1])),
                 showlegend=False)],
            rows=r, cols=c)
    fig.update_layout(
        autosize=False,
        margin=dict(l=20, r=20, t=20, b=20),
        title={
            'text': f"Decision surfaces of different number of classifiers ("
                    f"noise={noise})",
            'y': 1, 'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(size=10),
        width=800, height=800)
    fig.write_image(os.path.join(f"plot_ada_{noise}_decisions.png"))

    # Question 3: Decision surface of best performing ensemble
    t_min_loss = np.argmin(test_losses) + 1
    accuracy = 1 - round(test_losses[t_min_loss - 1], 2)
    surface_min = decision_surface(
        predict=(lambda X: ada_learner.partial_predict(X, t_min_loss)),
        xrange=lims[0], yrange=lims[1],  showscale=False, density=50)
    go.Figure([
        surface_min,
        go.Scatter(
            x=test_X[:, 0], y=test_X[:, 1],
            mode="markers",
            marker=dict(
                color=test_y,
                symbol=np.where(test_y == 1, class_symbols[0],
                                class_symbols[1])),
            showlegend=False)]
        ).update_layout(
        title=f"Surface of ensemble with minimum error<br>"
              f"(size={t_min_loss}, accuracy={accuracy}, noise={noise})",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        width=600, height=600, margin=dict(l=20, r=20, t=50, b=20)
        ).write_image(os.path.join(f"plot_ada_{noise}_min_err.png"))

    # Question 4: Decision surface with weighted samples
    D = ada_learner.D_ / ada_learner.D_.max() * 15
    surface_D = decision_surface(
        predict=ada_learner.predict,
        xrange=lims[0], yrange=lims[1], showscale=False)
    go.Figure([
        surface_D,
        go.Scatter(
            x=train_X[:, 0], y=train_X[:, 1],
            mode="markers",
            marker=dict(
                size=D,
                color=train_y,
                symbol=np.where(train_y == 1, class_symbols[0],
                                class_symbols[1])),
            showlegend=False)]
        ).update_layout(
        title=f"Distribution of Adaboost's last sample with weighted samples "
              f"(noise={noise})",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        width=600, height=600, margin=dict(l=20, r=20, t=50, b=20)
        ).write_image(os.path.join(f"plot_ada_{noise}_last.png"))


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
