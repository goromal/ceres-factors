# Ceres Factors

C++ library with custom parameterizations and cost functions for the Ceres Solver:

- *SO3LocalParameterization* (chart map implementation)
- *SE3LocalParameterization* (chart map implementation)
- *SO3Factor* (e.g., rotation averaging)
- *RelSE3Factor* (e.g., pose graph optimization)
- *RangeFactor* (for fusing point-to-point range measurements with pose measurements)
- *AltFactor* (for fusing altimeter measurements with pose measurements)
- *TimeSyncAttFactor* (for determining small time offsets by comparing attitude and Euler-integrated gyro measurements)

The Ceres Solver (http://ceres-solver.org/) is Google's powerful and extensive C++ optimization library for solving:

1. general unconstrained optimization problems
2. *nonlinear least-squares problems with bounds (not equality!) constraints*

The second application is particularly useful for perception, estimation, and control in robotics (perhaps less so for control, depending on how you implement dynamics constraints), where minimizing general nonlinear measurement or tracking residuals sequentially or in batch form is a common theme. This is further facilitated by Ceres' support for optimizing over vector spaces as well as Lie Groups

Advantages over some other open-source nonlinear least-squares solvers like [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares):

- Much faster
- Scales better to larger problems (used as Google's bundle adjustment backend)
- Has a built-in auto-differentiation engine that provides exact derivatives and is sometimes even *faster* than supplying analytic Jacobians.
- Has built-in support for optimizing over manifolds like SO(3)/SE(3).

## Dependencies

- ceres-solver
- Eigen3
- [manif-geom-cpp](https://github.com/goromal/manif-geom-cpp)
