#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <SO2.h>
#include <SE2.h>
#include <SO3.h>
#include <SE3.h>

using namespace Eigen;

// AutoDiff cost function (factor) for the difference between two rotations.
// Weighted by measurement covariance, Q_.
class SO3Factor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // store measured relative pose and inverted covariance matrix
    SO3Factor(const Vector4d& q_vec, const Matrix3d& Q) : q_(q_vec), Q_inv_(Q.inverse()) {}

    // templated residual definition for both doubles and jets
    // basically a weighted implementation of boxminus using Eigen templated types
    template<typename T>
    bool operator()(const T* _q_hat, T* _res) const
    {
        SO3<T>               q_hat(_q_hat);
        Map<Matrix<T, 3, 1>> r(_res);
        r = Q_inv_ * (q_hat - q_.cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const Vector4d& q_vec, const Matrix3d& Q)
    {
        return new ceres::AutoDiffCostFunction<SO3Factor, 3, 4>(new SO3Factor(q_vec, Q));
    }

private:
    SO3d     q_;
    Matrix3d Q_inv_;
};

// AutoDiff cost function (factor) for the difference between a measured 3D
// relative transform, Xij = (tij_, qij_), and the relative transform between two
// estimated poses, Xi_hat and Xj_hat. Weighted by measurement covariance, Qij_.
class RelSE3Factor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef Matrix<double, 7, 1> Vector7d;
    typedef Matrix<double, 6, 6> Matrix6d;
    // store measured relative pose and inverted covariance matrix
    RelSE3Factor(const Vector7d& X_vec, const Matrix6d& Q) : Xij_(X_vec), Q_inv_(Q.inverse()) {}

    // templated residual definition for both doubles and jets
    // basically a weighted implementation of boxminus using Eigen templated types
    template<typename T>
    bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
    {
        SE3<T>               Xi_hat(_Xi_hat);
        SE3<T>               Xj_hat(_Xj_hat);
        Map<Matrix<T, 6, 1>> r(_res);
        r = Q_inv_ * (Xi_hat.inverse() * Xj_hat - Xij_.cast<T>());
        return true;
    }

    static ceres::CostFunction* Create(const Vector7d& Xij, const Matrix6d& Q)
    {
        return new ceres::AutoDiffCostFunction<RelSE3Factor, 6, 7, 7>(new RelSE3Factor(Xij, Q));
    }

private:
    SE3d     Xij_;
    Matrix6d Q_inv_;
};

// AutoDiff cost function (factor) for the difference between a range measurement
// rij, and the relative range between two estimated poses, Xi_hat and Xj_hat.
// Weighted by measurement variance, qij_.
class RangeFactor
{
public:
    // store measured range and inverted variance
    RangeFactor(double& rij, double& qij)
    {
        rij_     = rij;
        qij_inv_ = 1.0 / qij;
    }

    // templated residual definition for both doubles and jets
    template<typename T>
    bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
    {
        Eigen::Matrix<T, 3, 1> ti_hat(_Xi_hat), tj_hat(_Xj_hat);
        *_res = static_cast<T>(qij_inv_) * (static_cast<T>(rij_) - (tj_hat - ti_hat).norm());
        return true;
    }

    // cost function generator--ONLY FOR PYTHON WRAPPER
    static ceres::CostFunction* Create(double& rij, double& qij)
    {
        return new ceres::AutoDiffCostFunction<RangeFactor, 1, 7, 7>(new RangeFactor(rij, qij));
    }

private:
    double rij_;
    double qij_inv_;
};

class RangeBearing2DFactor
{
public:
    /**
     * Initialize factor with reference measurements and covariances.
     *
     * @param d_k Range measurement.
     * @param sigma_d_k Noise associated with the range measurement.
     * @param theta_k Bearing measurement w.r.t. the vehicle body frame.
     * @param sigma_theta_k Noise associated with the bearing measurement.
     * @param p_k Position of the vehicle in the world frame.
     * @param phi_k Rotation of the vehicle w.r.t. the world frame.
     */
    RangeBearing2DFactor(const double&   d_k,
                         const double&   sigma_d_k,
                         const double&   theta_k,
                         const double&   sigma_theta_k,
                         const Vector2d& p_k,
                         const double&   phi_k)
        : p_k_(p_k)
    {
        // Construct sensor-to-world rotation
        SO2d R_S_W = SO2d::fromAngle(phi_k) * SO2d::fromAngle(theta_k);
        // Construct bearing vector
        Vector2d b_k_ = R_S_W * Vector2d(d_k, 0);
        // Construct rotated bearing covariance matrix
        Matrix2d Sigma_S =
            (Matrix2d() << sigma_d_k * sigma_d_k, 0, 0, d_k * d_k * sigma_theta_k * sigma_theta_k).finished();
        Sigma_k_inv_ = (R_S_W.R() * Sigma_S * R_S_W.R().transpose()).inverse();
    }

    /**
     * Templated residual function
     */
    template<typename T>
    bool operator()(const T* _l_hat, const T* _R_err_hat, T* _res) const
    {
        Eigen::Matrix<T, 2, 1> l_hat(_l_hat);
        SO2<T>                 R_err_hat(_R_err_hat);
        Map<Matrix<T, 2, 1>>   r(_res);
        r = Sigma_k_inv_ * (b_k_.cast<T>() - R_err_hat * (l_hat - p_k_.cast<T>()));
        return true;
    }

    // cost function generator--ONLY FOR PYTHON WRAPPER
    static ceres::CostFunction* Create(const double&   d_k,
                                       const double&   sigma_d_k,
                                       const double&   theta_k,
                                       const double&   sigma_theta_k,
                                       const Vector2d& p_k,
                                       const double&   phi_k)
    {
        return new ceres::AutoDiffCostFunction<RangeBearing2DFactor, 2, 2, 2>(
            new RangeBearing2DFactor(d_k, sigma_d_k, theta_k, sigma_theta_k, p_k, phi_k));
    }

private:
    Vector2d b_k_;
    Matrix2d Sigma_k_inv_;
    Vector2d p_k_;
};

// AutoDiff cost function (factor) for the difference between an altitude
// measurement hi, and the altitude of an estimated pose, Xi_hat.
// Weighted by measurement variance, qi_.
class AltFactor
{
public:
    // store measured range and inverted variance
    AltFactor(double& hi, double& qi)
    {
        hi_     = hi;
        qi_inv_ = 1.0 / qi;
    }

    // templated residual definition for both doubles and jets
    template<typename T>
    bool operator()(const T* _Xi_hat, T* _res) const
    {
        T hi_hat = *(_Xi_hat + 2);
        *_res    = static_cast<T>(qi_inv_) * (static_cast<T>(hi_) - hi_hat);
        return true;
    }

    // cost function generator--ONLY FOR PYTHON WRAPPER
    static ceres::CostFunction* Create(double& hi, double& qi)
    {
        return new ceres::AutoDiffCostFunction<AltFactor, 1, 7>(new AltFactor(hi, qi));
    }

private:
    double hi_;
    double qi_inv_;
};

// AutoDiff cost function (factor) for time-syncing attitude measurements,
// giving the residual q_ref - (q + dt * w), where dt is the decision
// variable. Weighted by measurement covariance, Q[3x3].
class TimeSyncAttFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TimeSyncAttFactor(const Vector4d& q_ref_vec, const Vector4d& q_vec, const Vector3d& w_vec, const Matrix3d& Q)
        : q_ref_(q_ref_vec), q_(q_vec), w_(w_vec), Q_inv_(Q.inverse())
    {
    }

    template<typename T>
    bool operator()(const T* _dt_hat, T* _res) const
    {
        Map<Matrix<T, 3, 1>> r(_res);
        r = Q_inv_ * (q_ref_.cast<T>() - (q_.cast<T>() + *_dt_hat * w_.cast<T>()));
        return true;
    }

    static ceres::CostFunction*
    Create(const Vector4d& q_ref_vec, const Vector4d& q_vec, const Vector3d& w_vec, const Matrix3d& Q)
    {
        return new ceres::AutoDiffCostFunction<TimeSyncAttFactor, 3, 1>(
            new TimeSyncAttFactor(q_ref_vec, q_vec, w_vec, Q));
    }

private:
    SO3d     q_ref_;
    SO3d     q_;
    Vector3d w_;
    Matrix3d Q_inv_;
};

// AutoDiff cost function (factor) for SO3 offset calibration from attitude measurements,
// giving the residual q_ref - (q * q_off), where q_off is the decision variable.
// Weighted by measurement covariance, Q[3x3].
class SO3OffsetFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SO3OffsetFactor(const Vector4d& q_ref_vec, const Vector4d& q_vec, const Matrix3d& Q)
        : q_ref_(q_ref_vec), q_(q_vec), Q_inv_(Q.inverse())
    {
    }

    template<typename T>
    bool operator()(const T* _q_off, T* _res) const
    {
        SO3<T>               q_off(_q_off);
        Map<Matrix<T, 3, 1>> r(_res);
        r = Q_inv_ * (q_ref_.cast<T>() - (q_.cast<T>() * q_off));
        return true;
    }

    static ceres::CostFunction* Create(const Vector4d& q_ref_vec, const Vector4d& q_vec, const Matrix3d& Q)
    {
        return new ceres::AutoDiffCostFunction<SO3OffsetFactor, 3, 4>(new SO3OffsetFactor(q_ref_vec, q_vec, Q));
    }

private:
    SO3d     q_ref_;
    SO3d     q_;
    Matrix3d Q_inv_;
};

// AutoDiff cost function (factor) for SE3 offset calibration from pose measurements,
// giving the residual T_ref - (T * T_off), where T_off is the decision variable.
// Weighted by measurement covariance, Q[6x6].
class SE3OffsetFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef Matrix<double, 7, 1> Vector7d;
    typedef Matrix<double, 6, 6> Matrix6d;
    SE3OffsetFactor(const Vector7d& T_ref_vec, const Vector7d& T_vec, const Matrix6d& Q)
        : T_ref_(T_ref_vec), T_(T_vec), Q_inv_(Q.inverse())
    {
    }

    template<typename T>
    bool operator()(const T* _T_off, T* _res) const
    {
        SE3<T>               T_off(_T_off);
        Map<Matrix<T, 6, 1>> r(_res);
        r = Q_inv_ * (T_ref_.cast<T>() - (T_.cast<T>() * T_off));
        return true;
    }

    static ceres::CostFunction* Create(const Vector7d& T_ref_vec, const Vector7d& T_vec, const Matrix6d& Q)
    {
        return new ceres::AutoDiffCostFunction<SE3OffsetFactor, 6, 7>(new SE3OffsetFactor(T_ref_vec, T_vec, Q));
    }

private:
    SE3d     T_ref_;
    SE3d     T_;
    Matrix6d Q_inv_;
};
