import os
import numpy as np

def generate_datasets() -> None:
    # Ensure test data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    np.random.seed(42)  # Set seed for reproducible test results
    
    # 1. Perfect Line (y = 3x + 5) - Ideal for exact math verification
    x1 = np.linspace(0, 10, 50)
    y1 = 3.0 * x1 + 5.0
    np.savetxt(os.path.join(data_dir, "perfect_line.csv"), np.column_stack((x1, y1)), delimiter=",", fmt="%.4f")

    # 2. Line with Gaussian Noise (y = -1.5x + 10 + noise) - Ideal for convergence testing
    x2 = np.linspace(-5, 5, 100)
    noise = np.random.normal(0, 1.5, size=x2.shape)
    y2 = -1.5 * x2 + 10.0 + noise
    np.savetxt(os.path.join(data_dir, "noisy_line.csv"), np.column_stack((x2, y2)), delimiter=",", fmt="%.4f")

    # 3. Horizontal Line (y = 4) - Slope m = 0 boundary test
    x3 = np.linspace(-10, 10, 40)
    y3 = np.full_like(x3, 4.0)
    np.savetxt(os.path.join(data_dir, "horizontal_line.csv"), np.column_stack((x3, y3)), delimiter=",", fmt="%.4f")

    print(f"Test datasets created inside '{data_dir}/'")

if __name__ == "__main__":
    generate_datasets()