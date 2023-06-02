#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <SO2.h>
#include <SE2.h>
#include <SO3.h>
#include <SE3.h>

// AutoDiff local parameterization for the SO2 rotation [q] object.
// The boxplus operator informs Ceres how the manifold evolves and also
// allows for the calculation of derivatives.
struct SO2Parameterization
{
    // boxplus operator for both doubles and jets
    template<typename T>
    bool operator()(const T* x, const T* delta, T* x_plus_delta) const
    {
        SO2<T>                                   X(x);
        Eigen::Map<const Eigen::Matrix<T, 1, 1>> dX(delta);
        Eigen::Map<Eigen::Matrix<T, 2, 1>>       Yvec(x_plus_delta);

        Yvec << (X + dX).array();

        return true;
    }

    static ceres::LocalParameterization* Create()
    {
        return new ceres::AutoDiffLocalParameterization<SO2Parameterization, 2, 1>();
    }
};

// AutoDiff local parameterization for the compact SE2 pose [t q] object.
// The boxplus operator informs Ceres how the manifold evolves and also
// allows for the calculation of derivatives.
struct SE2Parameterization
{
    // boxplus operator for both doubles and jets
    template<typename T>
    bool operator()(const T* x, const T* delta, T* x_plus_delta) const
    {
        SE2<T>                                   X(x);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> dX(delta);
        Eigen::Map<Eigen::Matrix<T, 4, 1>>       Yvec(x_plus_delta);

        Yvec << (X + dX).array();

        return true;
    }

    static ceres::LocalParameterization* Create()
    {
        return new ceres::AutoDiffLocalParameterization<SE2Parameterization, 4, 3>();
    }
};

// AutoDiff local parameterization for the SO3 rotation [q] object.
// The boxplus operator informs Ceres how the manifold evolves and also
// allows for the calculation of derivatives.
struct SO3Parameterization
{
    // boxplus operator for both doubles and jets
    template<typename T>
    bool operator()(const T* x, const T* delta, T* x_plus_delta) const
    {
        SO3<T>                                   X(x);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> dX(delta);
        Eigen::Map<Eigen::Matrix<T, 4, 1>>       Yvec(x_plus_delta);

        Yvec << (X + dX).array();

        return true;
    }

    static ceres::LocalParameterization* Create()
    {
        return new ceres::AutoDiffLocalParameterization<SO3Parameterization, 4, 3>();
    }
};

// AutoDiff local parameterization for the compact SE3 pose [t q] object.
// The boxplus operator informs Ceres how the manifold evolves and also
// allows for the calculation of derivatives.
struct SE3Parameterization
{
    // boxplus operator for both doubles and jets
    template<typename T>
    bool operator()(const T* x, const T* delta, T* x_plus_delta) const
    {
        SE3<T>                                   X(x);
        Eigen::Map<const Eigen::Matrix<T, 6, 1>> dX(delta);
        Eigen::Map<Eigen::Matrix<T, 7, 1>>       Yvec(x_plus_delta);

        Yvec << (X + dX).array();

        return true;
    }

    static ceres::LocalParameterization* Create()
    {
        return new ceres::AutoDiffLocalParameterization<SE3Parameterization, 7, 6>();
    }
};
