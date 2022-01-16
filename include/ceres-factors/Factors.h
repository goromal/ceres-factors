#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
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
  SO3Factor(const Vector4d &q_vec, const Matrix3d &Q)
  : q_(q_vec), Q_inv_(Q.inverse())
  {}

  // templated residual definition for both doubles and jets
  // basically a weighted implementation of boxminus using Eigen templated types
  template<typename T>
  bool operator()(const T* _q_hat, T* _res) const
  {
    SO3<T> q_hat(_q_hat);
    Map<Matrix<T,3,1>> r(_res);
    r = Q_inv_ * (q_hat - q_.cast<T>());
    return true;
  }

  static ceres::CostFunction *Create(const Vector4d &q_vec, const Matrix3d &Q) {
    return new ceres::AutoDiffCostFunction<SO3Factor,
                                           3,
                                           4>(new SO3Factor(q_vec, Q));
  }

private:
  SO3d q_;
  Matrix3d Q_inv_;
};

// AutoDiff cost function (factor) for the difference between a measured 3D
// relative transform, Xij = (tij_, qij_), and the relative transform between two  
// estimated poses, Xi_hat and Xj_hat. Weighted by measurement covariance, Qij_.
class RelSE3Factor
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Matrix<double,7,1> Vector7d;
  typedef Matrix<double,6,6> Matrix6d;
  // store measured relative pose and inverted covariance matrix
  RelSE3Factor(const Vector7d &X_vec, const Matrix6d &Q)
  : Xij_(X_vec), Q_inv_(Q.inverse())
  {}

  // templated residual definition for both doubles and jets
  // basically a weighted implementation of boxminus using Eigen templated types
  template<typename T>
  bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
  {
    SE3<T> Xi_hat(_Xi_hat);
    SE3<T> Xj_hat(_Xj_hat);
    Map<Matrix<T,6,1>> r(_res);
    r = Q_inv_ * (Xi_hat.inverse() * Xj_hat - Xij_.cast<T>());  
    return true;
  }

  static ceres::CostFunction *Create(const Vector7d &Xij, const Matrix6d &Q) {
    return new ceres::AutoDiffCostFunction<RelSE3Factor,
                                           6,
                                           7,
                                           7>(new RelSE3Factor(Xij, Q));
  }

private:
  SE3d Xij_;
  Matrix6d Q_inv_;
};

// AutoDiff cost function (factor) for the difference between a range measurement
// rij, and the relative range between two estimated poses, Xi_hat and Xj_hat. 
// Weighted by measurement variance, qij_.
class RangeFactor
{
public:
  // store measured range and inverted variance
  RangeFactor(double &rij, double &qij)
  {
    rij_ = rij;
    qij_inv_ = 1.0 / qij;
  }

  // templated residual definition for both doubles and jets
  template<typename T>
  bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
  {
    Eigen::Matrix<T,3,1> ti_hat(_Xi_hat), tj_hat(_Xj_hat);
    *_res = static_cast<T>(qij_inv_) * (static_cast<T>(rij_) - (tj_hat - ti_hat).norm());
    return true;
  }

  // cost function generator--ONLY FOR PYTHON WRAPPER
  static ceres::CostFunction *Create(double &rij, double &qij) {
    return new ceres::AutoDiffCostFunction<RangeFactor,
                                           1,
                                           7,
                                           7>(new RangeFactor(rij, qij));
  }

private:
  double rij_;
  double qij_inv_;
};

// AutoDiff cost function (factor) for the difference between an altitude 
// measurement hi, and the altitude of an estimated pose, Xi_hat. 
// Weighted by measurement variance, qi_.
class AltFactor
{
public:
  // store measured range and inverted variance
  AltFactor(double &hi, double &qi)
  {
    hi_ = hi;
    qi_inv_ = 1.0 / qi;
  }

  // templated residual definition for both doubles and jets
  template<typename T>
  bool operator()(const T* _Xi_hat, T* _res) const
  {
    T hi_hat = *(_Xi_hat + 2);
    *_res = static_cast<T>(qi_inv_) * (static_cast<T>(hi_) - hi_hat);
    return true;
  }

  // cost function generator--ONLY FOR PYTHON WRAPPER
  static ceres::CostFunction *Create(double &hi, double &qi) {
    return new ceres::AutoDiffCostFunction<AltFactor,
                                           1,
                                           7>(new AltFactor(hi, qi));
  }

private:
  double hi_;
  double qi_inv_;
};