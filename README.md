# Ceres Factors

![example workflow](https://github.com/goromal/ceres-factors/actions/workflows/test.yml/badge.svg)

C++ library with custom parameterizations and cost functions for the Ceres Solver:

- *SO3LocalParameterization* (chart map implementation)
- *SE3LocalParameterization* (chart map implementation)
- *SO3Factor* (e.g., rotation averaging)
- *RelSE3Factor* (e.g., pose graph optimization)
- *RangeFactor* (for fusing point-to-point range measurements with pose measurements)
- *AltFactor* (for fusing altimeter measurements with pose measurements)
- *TimeSyncAttFactor* (for determining small time offsets by comparing attitude and Euler-integrated gyro measurements)
- *SO3OffsetFactor* (for calibrating rotation offsets)
- *SE3OffsetFactor* (for calibrating pose offsets)
- *RangeBearing2DFactor* (for incorporating a range + bearing sensor in 2D)

Some explanations and illustrative examples are available in my public-facing subset of [notes on optimization libraries](https://notes.andrewtorgesen.com/doku.php?id=public:autonomy:implementation:opt-libs).

The Ceres Solver (http://ceres-solver.org/) is Google's powerful and extensive C++ optimization library for solving:

1. general unconstrained optimization problems
2. *nonlinear least-squares problems with bounds (not equality!) constraints*

The second application is particularly useful for perception, estimation, and control in robotics (perhaps less so for control, depending on how you implement dynamics constraints), where minimizing general nonlinear measurement or tracking residuals sequentially or in batch form is a common theme. This is further facilitated by Ceres' support for optimizing over vector spaces as well as Lie Groups

Advantages over some other open-source nonlinear least-squares solvers like [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares):

- Much faster
- Scales better to larger problems (used as Google's bundle adjustment backend)
- Has a built-in auto-differentiation engine that provides exact derivatives and is sometimes even *faster* than supplying analytic Jacobians.
- Has built-in support for optimizing over manifolds like SO(3)/SE(3).

## Building / Installing

This library is built with CMake. Most recently tested with the following dependencies:

- ceres-solver 2.0.0
- Eigen 3.4.0
- [manif-geom-cpp](https://github.com/goromal/manif-geom-cpp)
- Boost 1.79.0 (for unit test framework)

```bash
mkdir build
cd build
cmake ..
make # or make install
```

By default, building will build and run the unit tests, but this can be turned off with the CMake option `BUILD_TESTS`.
