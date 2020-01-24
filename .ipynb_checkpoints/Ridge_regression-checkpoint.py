# Authors: Cristina Aguilera, Jesus Antonanzas
#          GCED, OM - grup 11.
#          May 2019
#
# Python implementation of ridge - or Tikhonov - regularization:
#
# min          1/2(A*w + gamma - y)'*(A*w + gamma - y)
# subject to   w'*w <= t
#
# Where 'A' is a set of observations, 'gamma' a vector of real numbers
# and 'y' the target variable.

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

# =========== INPUT DATA ===========
# Read dataset.
# Observations by rows and variables by columns.
# Target variable needs to be the last column.

A = np.loadtxt('deathrate_instance_python.dat')

# Copy target variable.
y = A[:,len(A[0])-1]

# Delete target variable from dataset.
A = A[:,0:len(A[0])-1]

# Declare dimensions:
#   m = # of observations.
#   n = # of variables.
m = A.shape[0]
n = A.shape[1]

# =========== PRELIMINARY COMPUTATIONS ===========

# First, declare variables that remain constant
# and come in handy during calculations.

At = A.transpose()
AtA = np.matmul(At,A)
Aty = np.matmul(At,y)
ones_m = np.ones(m)
ones_dot_y = np.dot(ones_m,y)

# Gradient of the restriction:
upperleft = np.matmul(At,A)
upperright = np.matmul(At,np.ones((m,1)))
tophalf = np.append(upperleft, upperright, axis = 1)
bottomhalf = np.append(upperright,m)
const_grad_var = np.append(tophalf, [bottomhalf], axis = 0)

# Hessian of the constraint
cons_hess_var = np.diag(np.append(np.full((n),2),0))


# =========== EVALUATION FUNCTIONS ===========

def obj_f(w):
    # Pos 15 of w contains gamma, all other positions are w_i, i = 0...n-1 (num. var).
    gamma = w[n]
    w = w[0:n]
    # gamma = gamma * np.ones((m,1)) veure si funciona sense
    p1 = np.dot(A, w) + gamma * ones_m - y
    return 1/2 * np.dot(p1, p1)

def obj_grad(w):
    # Pos 16 of w contains gamma, all other positions are w_i, i = 0...n-1 (num. var).
    gamma = w[n]
    w = w[0:n]
    grad = np.zeros(n+1)
    AtAw = np.matmul(AtA,w)
    grad[:-1] = AtAw + np.matmul(At, gamma*ones_m) - Aty
    grad[-1] = np.dot(np.matmul(w.transpose(),At),ones_m) + m*gamma - ones_dot_y
    return grad

def obj_hess(w):
    # Returned value remains constant, so just calculate once at the initialisation.
    return const_grad_var

def cons(w):
    w = w[0:n]
    return np.dot(w.transpose(),w)

def cons_grad(w):
    return np.append(2*w[0:n],0)

def cons_hess(w,v):
    # Returns sum of all constraint Hessians times their respective Lagrange multipliers.
    return v[0]*cons_hess_var

# =========== MINIMIZATION ===========

# Declare initial values.
w0 = np.zeros(n+1)
t = 1

# Declare constraint.
nonlinear_constraint = NonlinearConstraint(cons, lb = -np.inf, ub = t,
                                           jac=cons_grad, hess=cons_hess)

# Variables are not bounded. We could write

# bounds = scipy.optimize.Bounds(-np.inf, np.inf)

# but increases computation time.

# Minimization.
solution = minimize(obj_f, w0, method='trust-constr', jac=obj_grad,
                    hess=obj_hess, constraints=[nonlinear_constraint],
                    options={'verbose': 1})

# Display the solution.
print(solution)
