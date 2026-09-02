# Linear Regression Project: Lessons Learnt & Reference Guide

This document captures key software engineering, testing, and machine learning concepts mastered while building a modular Linear Regression package from scratch.

---

## 1. Project Architecture & Execution

- **Package Structure:** Organizing code into dedicated `src/` (source code) and `tests/` (verification) directories creates clear boundaries between core functionality and test automation.

- **Package Discovery:** Placing an `__init__.py` file inside directories (e.g., `tests/__init__.py`) registers them as Python packages, allowing standard test runners (`unittest`, `pytest`) to discover modules across subdirectories.

- **Robust File Pathing:** Hardcoded relative paths break when tests are executed from different working directories. Using dynamic paths ensures reliability regardless of where the command is triggered:

```python
import os

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "sample.csv")
```

---

## 2. Machine Learning Core Concepts

- **Model Definition:** A model is a mathematical representation that maps inputs ($x$) to predictions ($\hat{y}$) using adjustable parameters/weights ($m$ for slope and $b$ for intercept).

- **Gradient Descent:** An optimization algorithm that iteratively adjusts parameters to minimize the Mean Squared Error (MSE) loss function. Gradient descent **is** the training process—it does **not** run after training.

- **Underfitting & Hyperparameter Tuning:** If a model fails to converge to the expected ground-truth parameters (e.g., producing $1.48x + 0.09$ instead of $2x + 1$), common causes include:

  - **Insufficient Iterations (`num_iterations`):** Optimization stopped before reaching the global minimum.
  - **Learning Rate ($\alpha$) Too Small:** Updates are tiny, requiring many more iterations to converge.
  - **Gradient Scale Differences:** The slope and intercept may update at different rates because the slope gradient depends on the input values.

---

## 3. Ground Truth Testing & Synthetic Data

- **Ground Truth Benchmarking:** Generating synthetic datasets from a known equation (e.g., $y = 2x + 1$) provides a deterministic baseline. Since the true values of $m$ and $b$ are known, unit tests can verify that gradient descent is implemented correctly.

- **Isolating Code Bugs from Data Noise:** Real-world datasets contain noise, missing values, and unscaled features. Testing first on clean synthetic data ensures failures are caused by implementation bugs rather than messy data.

- **Edge Case Verification:** Synthetic datasets make it easy to test scenarios such as:
  - Zero slope ($y = 4$)
  - Negative slopes
  - Controlled Gaussian noise

---

## 4. Testing Frameworks & Stages

- **Unit Testing (`unittest`):** Tests isolated components such as `predict()`, gradient updates, and file loaders.

- **Status Meaning:**
  - `OK` → All assertions passed.
  - `FAILED` → Assertion failures (`F`) or execution errors (`E`).

- **Automated Data Hooks:** Using

```python
@classmethod
def setUpClass(cls):
    ...
```

allows datasets to be generated automatically before test execution.

### Testing Stages Across the Model Lifecycle

1. **Before Training (Unit Testing)**
   - Verify code architecture.
   - Check parameter initialization.
   - Validate array shapes.
   - Confirm mathematical correctness.

2. **During Training (Validation & Monitoring)**
   - Monitor loss curves.
   - Ensure loss decreases over time.
   - Detect exploding or vanishing gradients.

3. **After Training (Evaluation)**
   - Evaluate on unseen test data.
   - Measure RMSE, MAE, and $R^2$.
   - Check for overfitting.

---

## 5. Overfitting & Model Generalization

- **Definition:** Overfitting occurs when a model memorizes training data instead of learning patterns that generalize.

### Typical Behavior

- **Training Loss:** Continues decreasing.
- **Validation Loss:** Begins increasing, indicating poorer performance on unseen data.

### Prevention Strategies

- **Early Stopping:** Stop training when validation loss stops improving.
- **Regularization ($L_1$ / $L_2$):** Penalize large weights to reduce model complexity.
- **Dataset Expansion:** Increase the amount of training data to improve generalization.