# **Numerical Analysis for Engineering - Assignment 2 - Block Coordinate Descent Method**

# **Costa Rica Institute of Technology**

**CE3102:** Numerical Analysis for Engineering  

Computer Engineering

Semester II - 2019

## **Assignment 2**

### **Block Coordinate Descent Method**

- This problem should be developed in Python.
- The task is to implement the iterative Block Coordinate Descent (BCD) method to find a solution to the optimization problem:

$$
  \min_{x \in \mathbb{R}^n} f(x)
$$

where \( f \) is a scalar function.

- Implement both the sequential and parallel versions of the BCD method to solve the optimization problem:

  \[
  \min_{x \in \mathbb{R}^{50}} \left( \frac{1}{2} x^T A x - b^T x \right),
  \]

where \( A \in \mathbb{R}^{50 \times 50} \) is a tridiagonal matrix and \( b \in \mathbb{R}^{50} \) such that:

  \[
  A = \begin{pmatrix}
  6 & 2 & 0 & 0 & \dots & 0 \\
  2 & 6 & 2 & 0 & \dots & 0 \\
  0 & 2 & 6 & 2 & \dots & 0 \\
  \vdots & \vdots & \ddots & \ddots & \ddots & \vdots \\
  0 & 0 & \dots & 2 & 6 & 2 \\
  0 & 0 & \dots & 0 & 2 & 6
  \end{pmatrix},
  \quad b = \begin{pmatrix}
  12 \\
  15 \\
  15 \\
  \vdots \\
  15 \\
  15 \\
  12
  \end{pmatrix}.
  \]

- In both implementations, the initial value should be \( x^{(0)} = (1, 1, \dots, 1)^T \in \mathbb{R}^{50} \). Additionally, the algorithm should stop when the condition \( \|\nabla f(x^{(k)})\| \leq 10^{-5} \) is met.
- The main file executing the sequential solution should be named `bcd_sequential.py`. The main file executing the parallel solution should be named `bcd_parallel.py`. Each file should generate an error plot of the number of iterations \( k \) versus the error \( \|\nabla f(x^{(k)})\| \).
