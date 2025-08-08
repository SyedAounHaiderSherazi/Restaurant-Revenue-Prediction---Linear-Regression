# 🍽️ Restaurant Revenue Prediction - Linear Regression

This project is part of Course 1: Supervised Machine Learning in the [Machine Learning Specialization by Andrew Ng (DeepLearning.AI)](https://www.coursera.org/specializations/machine-learning-introduction). The goal is to predict restaurant revenue based on population size using **Linear Regression**.

---

## 📌 Problem Statement

Given historical data of profits from restaurants in various cities and the populations of those cities, we aim to learn a linear relationship between **city population** (input feature) and **profit** (target value).

---

## 📈 Core Concepts & Techniques

### 🔁 Linear Regression Hypothesis Function:

\[
h_\theta(x) = \theta_0 + \theta_1x
\]

Where:
- \( x \) = population of a city  
- \( h_\theta(x) \) = predicted profit  
- \( \theta_0, \theta_1 \) = parameters learned via gradient descent

---

### 💰 Cost Function (Mean Squared Error):

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
\]

Where:
- \( m \) = number of training examples  
- \( y^{(i)} \) = actual profit  

---

### 🧠 Gradient Descent:

\[
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
\]

- \( \alpha \): learning rate  
- Iteratively updates parameters to minimize the cost function

---

## 🔧 Implementation Details

- Language: Python
- Libraries: NumPy, Matplotlib
- Gradient Descent implemented from scratch
- Loss function visualized to demonstrate convergence

---

## 📊 Output

- Predicted revenue vs. actual data plotted
- Cost function decreases over time (learning rate tuned for best convergence)
- Final model generalizes well for similar input data

---

## 📚 Skills Gained

- Understanding the math behind Linear Regression
- Implementing cost function and gradient descent
- Data visualization and loss plotting
