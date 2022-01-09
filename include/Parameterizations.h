#include <Eigen/Core>
#include <ceres/ceres.h>
#include <SO3.h>
#include <SE3.h>

// AutoDiff local parameterization for the SO3 rotation [q] object. 
// The boxplus operator informs Ceres how the manifold evolves and also  
// allows for the calculation of derivatives.
struct SO3Parameterization {
  // boxplus operator for both doubles and jets
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    SO3<T> X(x);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> dX(delta);
    
    SO3<T> Y = X + dX;
    
    Eigen::Map<Eigen::Matrix<T, 4, 1>> Yvec(x_plus_delta);
    Yvec << Y.array();

    return true;
  }
};

// AutoDiff local parameterization for the compact SE3 pose [t q] object. 
// The boxplus operator informs Ceres how the manifold evolves and also  
// allows for the calculation of derivatives.
struct SE3Parameterization {
  // boxplus operator for both doubles and jets
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    SE3<T> X(x);
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> dX(delta);
    
    SE3<T> Y = X + dX;
    
    Eigen::Map<Eigen::Matrix<T, 7, 1>> Yvec(x_plus_delta);
    Yvec << Y.array();

    return true;
  }
};