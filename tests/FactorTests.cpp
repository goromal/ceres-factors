#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <chrono>
#include <SO3.h>
#include <SE3.h>
#include <ceres/ceres.h>
#include "Factors.h"
#include "Parameterizations.h"

/*
NOTES:
- If Test*FactorResiduals passes but Test*FactorResidualsAndJacobians fails, then
  the factor implementation is not friendly to Jets.
*/

using namespace Eigen;

BOOST_AUTO_TEST_SUITE(TestFactors)

BOOST_AUTO_TEST_CASE(TestSO3FactorResiduals)
{
    Matrix3d Q = Matrix3d::Identity();
    SO3d q = SO3d::random();
    SO3d qhat = SO3d::identity();
    Vector3d qhat_minus_q = qhat - q;
    
    ceres::Problem problem;
    problem.AddParameterBlock(qhat.data(), 4, SO3Parameterization::Create());

    ceres::ResidualBlockId res_id = 
        problem.AddResidualBlock(SO3Factor::Create(q.array(), Q), 
                                 NULL,
                                 qhat.data());

    double cost;
    double res[3];
    problem.EvaluateResidualBlock(res_id, false, &cost, res, nullptr);

    BOOST_CHECK_CLOSE(res[0], qhat_minus_q.x(), 1e-8);
    BOOST_CHECK_CLOSE(res[1], qhat_minus_q.y(), 1e-8);
    BOOST_CHECK_CLOSE(res[2], qhat_minus_q.z(), 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3FactorResidualsAndJacobians)
{
    Matrix3d Q = Matrix3d::Identity();
    SO3d q = SO3d::random();
    SO3d qhat = SO3d::identity();
    Vector3d qhat_minus_q = qhat - q;
    
    ceres::Problem problem;
    problem.AddParameterBlock(qhat.data(), 4, SO3Parameterization::Create());

    ceres::ResidualBlockId res_id = 
        problem.AddResidualBlock(SO3Factor::Create(q.array(), Q), 
                                 NULL,
                                 qhat.data());

    double cost;
    double res[3];
    double jac1[3], jac2[3], jac3[3];
    double* jac[3] = {jac1, jac2, jac3};
    problem.EvaluateResidualBlock(res_id, false, &cost, res, jac);

    BOOST_CHECK_CLOSE(res[0], qhat_minus_q.x(), 1e-8);
    BOOST_CHECK_CLOSE(res[1], qhat_minus_q.y(), 1e-8);
    BOOST_CHECK_CLOSE(res[2], qhat_minus_q.z(), 1e-8);

    Matrix3d true_jacobian = Matrix3d::Identity();

    for (unsigned int i = 0; i < 3; ++i) 
        for (unsigned int j = 0; j < 3; ++j) 
            BOOST_CHECK_CLOSE(jac[i][j], true_jacobian(i,j), 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSE3Factor)
{
    // TODO

    // ceres::Solver::Options options;
    // options.max_num_iterations = 100;
    // options.num_threads = 4;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.minimizer_progress_to_stdout = true;
    // ceres::Solver::Summary summary;

    // ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << std::endl << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
