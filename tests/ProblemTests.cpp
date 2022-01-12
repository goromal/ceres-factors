#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <chrono>
#include <SO3.h>
#include <SE3.h>
#include <ceres/ceres.h>
#include "ceres-factors/Parameterizations.h"
#include "ceres-factors/Factors.h"

using namespace Eigen;

BOOST_AUTO_TEST_SUITE(TestAutoDiff)

BOOST_AUTO_TEST_CASE(TestSO3FactorProblem)
{
    srand(444444);
    Matrix3d Q = Matrix3d::Identity();
    SO3d q = SO3d::random();
    SO3d qhat = SO3d::identity();
    
    ceres::Problem problem;
    problem.AddParameterBlock(qhat.data(), 4, SO3Parameterization::Create());
    problem.AddResidualBlock(SO3Factor::Create(q.array(), Q),
                             nullptr,
                             qhat.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    BOOST_CHECK_CLOSE(q.w(), qhat.w(), 1e-4);
    BOOST_CHECK_CLOSE(q.x(), qhat.x(), 1e-4);
    BOOST_CHECK_CLOSE(q.y(), qhat.y(), 1e-4);
    BOOST_CHECK_CLOSE(q.z(), qhat.z(), 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()