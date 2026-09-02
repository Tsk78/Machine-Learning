import os
import unittest
import numpy as np
from numpy.typing import NDArray
from src.model import LinearRegressor

class TestLinearRegressor(unittest.TestCase):

    def setUp(self) -> None:
        """Set up a simple synthetic dataset: y = 2x + 1"""
        self.learning_rate = 0.01
        self.model = LinearRegressor(learning_rate=self.learning_rate)
        self.model.m = 0.0
        self.model.b = 0.0
        
        # Synthetic points generated from exact equation y = 2x + 1
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        self.points: NDArray[np.float64] = np.column_stack((x, y))

    def test_initialization(self) -> None:
        """Test default parameter initialization."""
        reg = LinearRegressor()
        self.assertEqual(reg.learning_rate, 0.0001)
        self.assertEqual(reg.m, 0.0)
        self.assertEqual(reg.b, 0.0)

    def test_predict(self) -> None:
        """Test prediction calculation for given x values."""
        self.model.m = 2.0
        self.model.b = 1.0
        x = np.array([1.0, 2.0, 3.0])
        predictions = self.model.predict(x)
        np.testing.assert_array_almost_equal(predictions, np.array([3.0, 5.0, 7.0]))

    def test_compute_error_zero(self) -> None:
        """Test MSE calculation when line perfectly fits the data."""
        self.model.m = 2.0
        self.model.b = 1.0
        error = self.model.compute_error(self.points)
        self.assertAlmostEqual(error, 0.0, places=5)

    def test_compute_error_nonzero(self) -> None:
        """Test MSE calculation when model has non-zero loss."""
        # Predictions will all be 0, y values are [3, 5, 7, 9, 11]
        # MSE = (3^2 + 5^2 + 7^2 + 9^2 + 11^2) / 5 = (9 + 25 + 49 + 81 + 121) / 5 = 57.0
        error = self.model.compute_error(self.points)
        self.assertAlmostEqual(error, 57.0, places=5)

    def test_fit_convergence(self) -> None:
        """Test whether gradient descent converges close to true parameters (m=2, b=1)."""
        m_fit, b_fit = self.model.fit(self.points, num_iterations=2000)
        
        # Verify fitted slope and intercept are within 0.1 of true values
        self.assertAlmostEqual(m_fit, 2.0, delta=0.1)
        self.assertAlmostEqual(b_fit, 1.0, delta=0.1)

    def test_fit_from_csv(self) -> None:
        """Verify fit performance using simple_line.csv data."""
        data_path = os.path.join(os.path.dirname(__file__), "data", "simple_line.csv")
        points: NDArray[np.float64] = np.genfromtxt(data_path, delimiter=",", dtype=np.float64)
        
        model = LinearRegressor(learning_rate=0.01)
        m_fit, b_fit = model.fit(points, num_iterations=2000)
        
        # Expect y = 2x + 1
        self.assertAlmostEqual(m_fit, 2.0, delta=0.05)
        self.assertAlmostEqual(b_fit, 1.0, delta=0.05)

if __name__ == "__main__":
    unittest.main()