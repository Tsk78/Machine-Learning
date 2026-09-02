"""Main script to run Linear Regression training on data.csv."""

import numpy as np
from src.model import LinearRegressor
from src.visualize import plot_regression


def main() -> None:
    # 1. Load dataset from CSV
    data_path = "data/data.csv"
    points: np.ndarray = np.genfromtxt(data_path, delimiter=",")
    print(f"Loaded {len(points)} data points from {data_path}.")

    # 2. Hyperparameters
    learning_rate: float = 0.0001
    num_iterations: int = 1000

    # 3. Instantiate model
    model = LinearRegressor(learning_rate=learning_rate)

    # 4. Initial metrics
    initial_mse: float = model.compute_error(points)
    print(f"Initial parameters: b = {model.b:.4f}, m = {model.m:.4f}")
    print(f"Initial Mean Squared Error: {initial_mse:.4f}")

    # 5. Train model
    print("\nRunning gradient descent...")
    b_opt, m_opt = model.fit(points, num_iterations=num_iterations)
    final_mse: float = model.compute_error(points)

    # 6. Final metrics
    print(f"Optimized parameters: b = {b_opt:.4f}, m = {m_opt:.4f}")
    print(f"Final Mean Squared Error: {final_mse:.4f}")

    # 7. Render plot
    plot_regression(points, m_opt, b_opt, final_mse)


if __name__ == "__main__":
    main()