#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <chrono>
#include <SO3.h>
#include <SE3.h>
#include <ceres/ceres.h>
#include "Factors.h"
#include "Parameterizations.h"

using namespace Eigen;

BOOST_AUTO_TEST_SUITE(TestFactors)

BOOST_AUTO_TEST_CASE(TestSO3Factor)
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
