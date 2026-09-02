# Linear Regression from Scratch: A Mathematical & Practical Guide

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%20%7C%20Matplotlib-orange.svg)

A modern, refactored, and pedagogical implementation of **Linear Regression using Gradient Descent from Scratch in Python**. 

This repository expands upon classic educational code (originally presented by Siraj Raval and Matt Nedrich) by introducing modern Python 3 idioms, object-oriented design, vectorized NumPy computation, interactive visualization tools, and clean LaTeX mathematical explanations.

---

## 📌 Project Overview

Instead of treating Machine Learning models as black boxes via high-level libraries like `scikit-learn` or `PyTorch`, this repository demonstrates how linear models optimize parameters directly from first principles.

We analyze a classic dataset: predicting **Student Test Scores** based on **Hours Spent Studying**.

### Key Educational Highlights
- **Mathematical Transparency:** Complete derivation of the loss surface and partial derivatives.
- **Vectorized NumPy Implementation:** Replaces slow explicit loops with scalable matrix calculus.
- **Interactive Notebooks:** Step-by-step visual exploration of parameter updates across iterations.
- **Modern Project Layout:** PEP 8 structured Python module complete with type annotations and `pytest` test suites.

---

## 📐 Mathematical Intuition

### 1. Hypothesis Function
Our model estimates target value $\hat{y}$ given input $x$ using the slope-intercept form:

$$\hat{y}_i = m \cdot x_i + b$$

Where:
- $m$: Slope / Weight
- $b$: Intercept / Bias
- $x_i$: Hours Studied for student $i$
- $\hat{y}_i$: Predicted Test Score

---

### 2. Loss Function (Mean Squared Error)
To quantify model performance across $N$ data points, we calculate the Mean Squared Error (MSE):

$$E(m, b) = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - (m \cdot x_i + b) \right)^2$$

---

### 3. Optimization via Gradient Descent
Gradient Descent updates $m$ and $b$ iteratively in the direction of steepest loss reduction by evaluating partial derivatives of $E(m, b)$:

$$\frac{\partial E}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} \left( y_i - (m \cdot x_i + b) \right)$$

$$\frac{\partial E}{\partial m} = -\frac{2}{N} \sum_{i=1}^{N} x_i \left( y_i - (m \cdot x_i + b) \right)$$

---

### 4. Parameter Updates
At each step, parameters move in opposition to the calculated gradient scaled by learning rate $\alpha$:

$$b \leftarrow b - \alpha \cdot \frac{\partial E}{\partial b}$$

$$m \leftarrow m - \alpha \cdot \frac{\partial E}{\partial m}$$

---

## 📁 Repository Structure

```text
linear-regression-from-scratch/
│
├── data/
│   └── data.csv                # Student test score dataset
├── notebooks/
│   └── linear_regression.ipynb # Interactive tutorial & visual walkthrough
├── src/
│   ├── __init__.py
│   ├── model.py                # Clean, documented LinearRegressor class
│   └── visualize.py            # Plotting loss curves & decision boundaries
├── tests/
│   ├── __init__.py
│   ├── generate_test_data.py   # Generates test data for unit testing
│   └── test_model.py           # Unit tests verifying gradient descent convergence
├── main.py                     # Entry point execution script
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT License
├── LESSONS_LEARNT.md           # Covers knowledge and tidbits gained 
└── README.md                   # Detailed learning guide