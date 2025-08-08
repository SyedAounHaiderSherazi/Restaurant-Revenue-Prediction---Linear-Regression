# 🍽️ Restaurant Revenue Prediction - Linear Regression

<img width="1390" height="602" alt="Screenshot 2025-08-08 050156" src="https://github.com/user-attachments/assets/054553c5-4753-43ea-8837-9398125ecf3b" />

<img width="1479" height="688" alt="Screenshot 2025-08-08 050251" src="https://github.com/user-attachments/assets/0a19a1f9-60bc-487d-bf6b-db74e6f15c74" />


This project is part of Course 1 of the Machine Learning Specialization by Andrew Ng (DeepLearning.AI). The goal is to predict restaurant revenue based on population size using **Linear Regression**.

---

## 📌 Problem Statement

Given data on city populations and corresponding restaurant profits, the objective is to learn a linear function that predicts profit from population.

---

## 📈 Concepts & Formulas

### Linear Hypothesis Function:

    h(x) = θ0 + θ1 * x

Where:
- x = population of the city
- h(x) = predicted profit
- θ0, θ1 = model parameters

---

### Cost Function (Mean Squared Error):

    J(θ) = (1 / (2 * m)) * Σ (h(xᶦ) - yᶦ)²

Where:
- m = number of training examples
- yᶦ = actual profit

---

### Gradient Descent Update Rule:

    θj := θj - α * (1 / m) * Σ ((h(xᶦ) - yᶦ) * xⱼᶦ)

- α = learning rate
- j = 0 or 1 depending on the parameter being updated

---

## 🔧 Implementation

- Language: Python
- Libraries: NumPy, Matplotlib
- Implemented cost function and gradient descent from scratch
- Visualized the learning process using loss plots

---

## 📊 Outputs

- Line of best fit on training data
- Cost function convergence graph
- Trained model for predictions on new city data

---

## 🧠 Skills Learned

- Linear regression theory and math
- Optimization with gradient descent
- Loss visualization and learning rate tuning
