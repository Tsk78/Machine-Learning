"""Visualization utilities for Linear Regression."""

import matplotlib.pyplot as plt
import numpy as np


def plot_regression(points: np.ndarray, m: float, b: float, mse: float) -> None:
    """Plots the raw data points alongside the fitted linear regression line.

    Args:
        points (np.ndarray): 2D array of shape (N, 2) with X and Y values.
        m (float): Fitted slope.
        b (float): Fitted y-intercept.
        mse (float): Final Mean Squared Error.
    """
    x: np.ndarray = points[:, 0]
    y: np.ndarray = points[:, 1]

    plt.figure(figsize=(8, 6))
    
    # Scatter plot of raw dataset
    plt.scatter(x, y, color="blue", alpha=0.6, label="Data Points")

    # Generate regression line
    x_line: np.ndarray = np.linspace(min(x), max(x), 100)
    y_line: np.ndarray = m * x_line + b
    plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Fit: y = {m:.2f}x + {b:.2f}")

    plt.title(f"Linear Regression Fit (MSE: {mse:.2f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()