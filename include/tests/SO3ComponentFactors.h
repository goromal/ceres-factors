#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <SO3.h>
#include <SE3.h>

using namespace Eigen;

class SO3ConstructorFactor
{
public:
    SO3ConstructorFactor() {}
    template<typename T>
    bool operator()(const T*_qhat, T* _res) const
    {
        SO3<T> q_hat(_qhat);
        _res[0] = q_hat.w();
        _res[1] = q_hat.x();
        _res[2] = q_hat.y();
        _res[3] = q_hat.z();
        return true;
    }
    static ceres::CostFunction *Create() {
        return new ceres::AutoDiffCostFunction<SO3ConstructorFactor, 4, 4>(new SO3ConstructorFactor());
    }
};

class SO3ResMapFactor
{
public:
    SO3ResMapFactor() {}
    template<typename T>
    bool operator()(const T*_qhat, T* _res) const
    {
        SO3<T> q_hat(_qhat);
        Map<Matrix<T,4,1>> r(_res);
        r = q_hat.array();
        return true;
    }
    static ceres::CostFunction *Create() {
        return new ceres::AutoDiffCostFunction<SO3ResMapFactor, 4, 4>(new SO3ResMapFactor());
    }
};

class SO3TrivialOMinusFactor
{
public:
    SO3TrivialOMinusFactor() {}
    template<typename T>
    bool operator()(const T*_qhat, T* _res) const
    {
        SO3<T> q_hat(_qhat);
        Map<Matrix<T,3,1>> r(_res);
        r = q_hat - q_hat;
        return true;
    }
    static ceres::CostFunction *Create() {
        return new ceres::AutoDiffCostFunction<SO3TrivialOMinusFactor, 3, 4>(new SO3TrivialOMinusFactor());
    }
};

class SO3CastFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SO3CastFactor(const Vector4d &q_vec) 
    : q_(q_vec), tmp_(4.0)
    {
        std::cout << "INIT: q_vec = " << q_vec.transpose() << std::endl;
        std::cout << "INIT: q_ = " << q_ << std::endl;
        std::cout << "INIT: q_.w() = " << q_.w() << std::endl;
        std::cout << "INIT: tmp_ = " << tmp_ << std::endl;
    }
    template<typename T>
    bool operator()(const T* _qhat, T* _res) const
    {
        std::cout << "q_.w() = " << q_.w() << std::endl;
        std::cout << "tmp_ = " << tmp_ << std::endl;
        _res[0] = _qhat[0] * q_.cast<T>().w();
        _res[1] = _qhat[0];
        return true;
    }
    static ceres::CostFunction *Create(const Vector4d &q_vec) {
        return new ceres::AutoDiffCostFunction<SO3CastFactor, 2, 4>(new SO3CastFactor(q_vec));
    }
private:
    SO3d q_;
    double tmp_;
};

class SO3OMinusFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SO3OMinusFactor(const Vector4d &q_vec) 
    : q_(q_vec)
    {}
    template<typename T>
    bool operator()(const T* _qhat, T* _res) const
    {
        SO3<T> q_hat(_qhat);
        Map<Matrix<T,3,1>> r(_res);
        r = q_hat - q_.cast<T>();
        return true;
    }
    static ceres::CostFunction *Create(const Vector4d &q_vec) {
        return new ceres::AutoDiffCostFunction<SO3OMinusFactor, 3, 4>(new SO3OMinusFactor(q_vec));
    }
private:
    SO3d q_;
};
