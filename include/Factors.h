#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <SO3.h>
#include <SE3.h>

using namespace Eigen;

// AutoDiff cost function (factor) for the difference between two rotations.
// Weighted by measurement covariance, Qij_.
class SO3Factor
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // store measured relative pose and inverted covariance matrix
  SO3Factor(const Eigen::Matrix<double,4,1> &qi_vec, const Eigen::Matrix<double,3,3> &Qij)
  : q_(qi_vec)
  {
    Qij_inv_ = Qij.inverse();
  }

  // templated residual definition for both doubles and jets
  // basically a weighted implementation of boxminus using Eigen templated types
  template<typename T>
  bool operator()(const T* _q_hat, T* _res) const
  {
    SO3<T> q_hat(_q_hat);
    Eigen::Map<Eigen::Matrix<T,3,1>> r(_res);
    
    r = Qij_inv_.cast<T>() * (q_hat - q_.cast<T>());
    
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Matrix<double,4,1> &qi, const Eigen::Matrix<double,3,3> &Qij) {
    return new ceres::AutoDiffCostFunction<SO3Factor,
                                           3,
                                           4>(new SO3Factor(qi, Qij));
  }

private:
  SO3d q_;
  Eigen::Matrix<double,3,3> Qij_inv_;
};

// AutoDiff cost function (factor) for the difference between a measured 3D
// relative transform, Xij = (tij_, qij_), and the relative transform between two  
// estimated poses, Xi_hat and Xj_hat. Weighted by measurement covariance, Qij_.
class SE3Factor
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // store measured relative pose and inverted covariance matrix
  SE3Factor(const Eigen::Matrix<double,7,1> &Xij_vec, const Eigen::Matrix<double,6,6> &Qij) : Xij_(Xij_vec)
  {
    Qij_inv_ = Qij.inverse();
  }

  // templated residual definition for both doubles and jets
  // basically a weighted implementation of boxminus using Eigen templated types
  template<typename T>
  bool operator()(const T* _Xi_hat, const T* _Xj_hat, T* _res) const
  {
    SE3<T> Xi_hat(_Xi_hat);
    SE3<T> Xj_hat(_Xj_hat);
    Eigen::Map<Eigen::Matrix<T,6,1>> r(_res);
    
    r = Qij_inv_.cast<T>() * (Xi_hat.inverse() * Xj_hat - Xij_.cast<T>());  

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Matrix<double,7,1> &Xij, const Eigen::Matrix<double,6,6> &Qij) {
    return new ceres::AutoDiffCostFunction<SE3Factor,
                                           6,
                                           7,
                                           7>(new SE3Factor(Xij, Qij));
  }

private:
  SE3d Xij_;
  Eigen::Matrix<double,6,6> Qij_inv_;
};
