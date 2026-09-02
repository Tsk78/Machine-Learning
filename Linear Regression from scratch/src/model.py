"""Linear Regression implementation using Gradient Descent from scratch."""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

class LinearRegressor:
    """Single-variable Linear Regression optimized via Gradient Descent.

    Fits a line of the form: y = m * x + b

    Attributes:
        learning_rate (float): Step size factor alpha for gradient descent updates.
        b (float): Intercept parameter (bias).
        m (float): Slope parameter (weight).
    """

    def __init__(self, learning_rate: float = 0.0001) -> None:
        """Initializes the regressor with default zero weights and learning rate."""
        self.learning_rate: float = learning_rate
        self.b: float = 0.0  # y-intercept
        self.m: float = 0.0  # slope
        self.cost_history: list[float] = []  # Store loss values here
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Computes predicted output y_hat for input array x.

        Args:
            x (np.ndarray): 1D array of feature values.

        Returns:
            np.ndarray: Predicted target values (m * x + b).
        """
        return self.m * x + self.b

    def compute_error(self, points: NDArray[np.float64]) -> float:
        """Calculates Mean Squared Error (MSE) over the dataset.

        Formula:
            MSE = (1 / N) * sum((y_i - (m * x_i + b))^2)

        Args:
            points (np.ndarray): 2D array of shape (N, 2) where column 0 is X
              and column 1 is Y.

        Returns:
            float: Calculated Mean Squared Error loss.
        """
        x: NDArray[np.float64] = points[:, 0]
        y: NDArray[np.float64] = points[:, 1]
        predictions = self.predict(x)
        errors = y - predictions
        return float(np.mean(errors ** 2))

    def step_gradient(self, points: NDArray[np.float64]) -> None:
        """Executes a single parameter update step using partial derivatives.

        Partial Derivatives w.r.t b and m:
            dL/db = -(2 / N) * sum(y - (m * x + b))
            dL/dm = -(2 / N) * sum(x * (y - (m * x + b)))

        Args:
            points (np.ndarray): 2D array of shape (N, 2) containing feature/target data.
        """
        x: NDArray[np.float64] = points[:, 0]
        y: NDArray[np.float64] = points[:, 1]
        N = float(len(points))

        # Vectorized calculation of prediction errors
        errors = y - self.predict(x)

        # Gradient computations
        b_gradient = -2/N * float(np.sum(errors))
        m_gradient = -2/N * float(np.sum(x * errors))

        # Gradient descent parameter updates
        self.b -= self.learning_rate * b_gradient
        self.m -= self.learning_rate * m_gradient

    def fit(self, points: NDArray[np.float64], num_iterations: int = 1000) -> Tuple[float, float]:
        """Runs the complete gradient descent optimization loop.

        Args:
            points (np.ndarray): 2D array of shape (N, 2) containing dataset points.
            num_iterations (int): Total number of gradient descent update steps.

        Returns:
            Tuple[float, float]: Converged parameter pair (intercept b, slope m).
        """
        self.cost_history = []  # Reset history prior to fitting
        for _ in range(num_iterations):
            # Track cost before/after the gradient step
            current_cost = self.compute_error(points)
            self.cost_history.append(current_cost)
                    
            self.step_gradient(points)  
                    
        return self.m, self.b