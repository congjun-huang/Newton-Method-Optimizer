# Newton-Method-Optimizer
## Step 1 - Implementing Newton’s Method optimizer

In this step I will write a python function, `newton`, which
implements Newton’s method for minimization of functions using PyTorch,
specificially using the autograd functionality.

### Specification:

The function signature is as follows,

    def newton(theta, f, tol = 1e-8, fscale=1.0, maxit = 100, max_half = 20)

with the arguments defined as follows:

-   `theta` is a vector of initial values for the optimization
    parameters.
-   `f` is the objective function to minimize. This function should take
    PyTorch tensors as inputs and returns a Tensor.
-   `tol` the convergence tolerance.
-   `fscale` a rough estimate of the magnitude of `f` at the optimum -
    used for convergence testing.
-   `maxit` the maximum number of Newton iterations to try before giving
    up.
-   `max_half` the maximum number of times a step should be halved
    before concluding that the step has failed to improve the objective.

My `newton` function will return a dictionary containing:

-   `f` the value of the objective function at the minimum.
-   `theta` the value of the parameters at the minimum.
-   `iter` the number of iterations taken to reach the minimum.
-   `grad` the gradient vector at the minimum (so the user can judge
    closeness to numerical zero).

The function will issue errors or warnings in at least the following
cases:

1.  If the objective or derivatives are not finite at the initial
    `theta`.
2.  If the step fails to reduce the objective despite trying `max_half`
    step halvings.
3.  If `maxit` is reached without convergence.
4.  If the Hessian is not positive definite at convergence.

### Other considerations:

1.  I will use `torch.double` type Tensors within my objective function(s) to
    maintain maximum precision.

2.  All gradient / jacobian and hessian values will be calculated
    using
    [`torch.autograd.functional.jacobian()`](https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html#torch-autograd-functional-jacobian)
    and
    [`torch.autograd.functional.hessian()`](https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html#torch-autograd-functional-hessian)
    respectively.

3.  I can test whether my Hessian is positive definite by seeing if
    `torch.linalg.cholesky()` succeeds in finding its Cholesky factor.
    If the Hessian is not positive definite, add a small multiple of the
    identity matrix to it and try again. One approach is to start by
    adding εI, where ε is the largest absolute value in the Hessian
    multiplied by 10<sup>-8</sup>. If the perturbed Hessian is still not
    positive definite, keep multiplying ε by 10 until it is.

4.  If the Newton step does not reduce the objective, or leads to a
    non-finite objective or derivatives, I will repeatedly
    half the step until the objective is reduced or `max_half` is
    reached.

5.  To judge whether the gradient vector is close enough to zero, I
    will consider the magnitude of the objective. The gradient can be judged to
    be close enough to zero when they are smaller than `tol` multiplied
    by the objective value. To handle the case where the objective value
    is close to 0 we add `fscale` to the objective value before
    multiplying by `tol`. Therefore, if `f0` and `g` are the current
    values of the objective function and gradient respectively then,
    `max(abs(g)) < (abs(f0)+fscale)*tol` is a suitable condition for
    convergence.

## Step 2 - Minimization examples

For each of the following minimization problems, I will use my `newton()`
function to find the minima using **at least 3 different** theta
starting values.

### 1. Quadratic function

Minimize the following function:

![f(x,y) = x^2-2x+2y^2+y+3](https://latex.codecogs.com/svg.latex?f%28x%2Cy%29%20%3D%20x%5E2-2x%2B2y%5E2%2By%2B3 "f(x,y) = x^2-2x+2y^2+y+3")

### 2. Rosenbrock’s function

Minimize the following function:

![f(x,y) = 10\*(y-x^2)^2 + (1-x)^2](https://latex.codecogs.com/svg.latex?f%28x%2Cy%29%20%3D%2010%2A%28y-x%5E2%29%5E2%20%2B%20%281-x%29%5E2 "f(x,y) = 10*(y-x^2)^2 + (1-x)^2")

### 3. Poisson regression likelihood

Assume the following data were observed,

``` python
x = [
   0.11, -0.06, -0.96, -0.48, -0.59, -0.42, -0.15,  1.14, 0.94, 
  -0.86, -0.08,  1.00, -2.01,  2.17, -0.20,  0.82, -0.13, 0.26, 
   0.22,  1.05
]

y = [4, 2, 4, 1, 1, 3, 4, 5, 7, 3, 5, 7, 0, 4, 2, 7, 3, 3, 2, 8]
```

fit a Poisson regression model to these data and estimate the values of
![\beta_0](https://latex.codecogs.com/svg.latex?%5Cbeta_0 "\beta_0") and
![\beta_1](https://latex.codecogs.com/svg.latex?%5Cbeta_1 "\beta_1") via
maximum likelihood.

The likelihood / log-likelihood function for Poisson regression is given
by,

![\begin{aligned}
\log \lambda_i &= \beta_0 + \beta_1 x_i \\\\
L(\beta_0,\beta_1) &= \prod\_{i=1}^{10}\frac{\lambda_i^{y_i} e^{-\lambda_i}}{y_i!} \\\\
l(\beta_0,\beta_1) &= \sum\_{i=1}^{10} y_i \log \lambda_i - \lambda_i - \log y_i!
\end{aligned}](https://latex.codecogs.com/svg.latex?%5Cbegin%7Baligned%7D%0A%5Clog%20%5Clambda_i%20%26%3D%20%5Cbeta_0%20%2B%20%5Cbeta_1%20x_i%20%5C%5C%0AL%28%5Cbeta_0%2C%5Cbeta_1%29%20%26%3D%20%5Cprod_%7Bi%3D1%7D%5E%7B10%7D%5Cfrac%7B%5Clambda_i%5E%7By_i%7D%20e%5E%7B-%5Clambda_i%7D%7D%7By_i%21%7D%20%5C%5C%0Al%28%5Cbeta_0%2C%5Cbeta_1%29%20%26%3D%20%5Csum_%7Bi%3D1%7D%5E%7B10%7D%20y_i%20%5Clog%20%5Clambda_i%20-%20%5Clambda_i%20-%20%5Clog%20y_i%21%0A%5Cend%7Baligned%7D "\begin{aligned}
\log \lambda_i &= \beta_0 + \beta_1 x_i \\
L(\beta_0,\beta_1) &= \prod_{i=1}^{10}\frac{\lambda_i^{y_i} e^{-\lambda_i}}{y_i!} \\
l(\beta_0,\beta_1) &= \sum_{i=1}^{10} y_i \log \lambda_i - \lambda_i - \log y_i!
\end{aligned}")

