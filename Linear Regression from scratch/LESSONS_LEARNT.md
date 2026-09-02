
---

```markdown
# Linear Regression Project: Lessons Learnt & Reference Guide

This document captures key software engineering, testing, and machine learning concepts mastered while building a modular Linear Regression package from scratch.

---

## 1. Project Architecture & Execution
* **Package Structure:** Organizing code into dedicated `src/` (source code) and `tests/` (verification) directories creates clear boundaries between core functionality and test automation.
* **Package Discovery:** Placing an `__init__.py` file inside directories (e.g., `tests/__init__.py`) registers them as Python packages, allowing standard test runners (`unittest`, `pytest`) to discover modules across subdirectories.
* **Robust File Pathing:** Hardcoded relative paths break when tests are executed from different working directories. Using dynamic paths ensures reliability regardless of where the command is triggered:
  ```python
  import os
  base_dir = os.path.dirname(__file__)
  data_path = os.path.join(base_dir, "data", "sample.csv")

```

---

## 2. Machine Learning Core Concepts

* **Model Definition:** A model is a mathematical representation that maps inputs ($x$) to predictions ($\hat{y}$) using adjustable parameters/weights ($m$ slope, $b$ intercept).
* **Gradient Descent:** An optimization algorithm that iteratively adjusts parameters to minimize the Mean Squared Error (MSE) loss function. It does *not* run after training—gradient descent *is* the process that trains the model.
* **Underfitting & Hyperparameter Tuning:** If a model fails to converge to expected ground-truth parameters (e.g., yielding $1.48x + 0.09$ instead of $2x + 1$), it is typically due to:
* **Insufficient Iterations (`num_iterations`):** Optimization was halted before reaching the global loss minimum.
* **Learning Rate ($\alpha$) Too Small:** Parameter update steps are too tiny, requiring significantly more iterations to converge.
* **Gradient Scale Differences:** Features and intercept terms receive updates at different rates because gradients for $m$ scale directly with input values $x$.



---

## 3. Ground Truth Testing & Synthetic Data

* **Ground Truth Benchmarking:** Generating synthetic datasets with a known target equation (e.g., $y = 2x + 1$) provides a deterministic baseline. Because the true $m$ and $b$ values are known in advance, unit tests can verify whether the gradient descent implementation is mathematically sound.
* **Isolating Code Bugs from Data Noise:** Real-world datasets introduce noise, missing values, and unscaled features. Testing first against clean, fake data ensures that failures stem strictly from mathematical or algorithmic bugs, rather than dirty data.
* **Edge Case Verification:** Synthetic datasets allow testing boundary conditions, such as zero slope ($y = 4$), negative slopes, or controlled Gaussian noise levels.

---

## 4. Testing Frameworks & Stages

* **Unit Testing (`unittest`):** Tests individual, isolated components (e.g., `predict()`, gradient updates, file loaders).
* **Status Meaning:** `OK` indicates all assertions passed without uncaught exceptions. `FAILED` indicates assertion mismatches (`F`) or execution errors (`E`).
* **Automated Data Hooks:** Using `@classmethod def setUpClass(cls)` allows tests to generate necessary dataset files dynamically before individual test cases execute.


* **Testing Stages Across Model Lifecycle:**
1. **Before Training (Unit Testing):** Verifies code architecture, initial parameter setups, array shape compatibility, and mathematical calculations.
2. **During Training (Validation & Monitoring):** Tracks loss curves to ensure monotonic loss reduction and checks for explosive or vanishing gradients.
3. **After Training (Evaluation & Overfitting Checks):** Measures performance on unseen holdout test sets using metrics like RMSE, MAE, and $R^2$.



---

## 5. Overfitting & Model Generalization

* **Definition:** Overfitting occurs when a model memorizes noise and specific quirks of the training data rather than learning general patterns.
* **Divergence Behavior:**
* **Training Loss:** Continues to decrease toward zero.
* **Validation Loss:** Begins to rise as performance on new, unseen data degrades.


* **Prevention Strategies:**
* **Early Stopping:** Halt training as soon as validation loss stops improving and begins to increase.
* **Regularization ($L_1 / L_2$):** Penalize large weight values to constrain model complexity.
* **Dataset Expansion:** Increase training sample size to force the learning of broader statistical trends.




