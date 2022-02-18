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

BOOST_AUTO_TEST_CASE(TestRelSE3FactorProblem)
{
    srand(444444);
    Matrix<double,6,6> Q = Matrix<double,6,6>::Identity();
    SE3d T0 = SE3d::identity();
    Matrix<double,6,1> wij;
    wij.setRandom();
    SE3d T1 = T0 + wij;
    SE3d Tij = SE3d::Exp(wij);
    SE3d T0hat = SE3d::identity();
    SE3d T1hat = SE3d::identity();

    ceres::Problem problem;
    problem.AddParameterBlock(T0hat.data(), 7, SE3Parameterization::Create());
    problem.SetParameterBlockConstant(T0hat.data());
    problem.AddParameterBlock(T1hat.data(), 7, SE3Parameterization::Create());
    problem.AddResidualBlock(RelSE3Factor::Create(Tij.array(), Q),
                             nullptr,
                             T0hat.data(),
                             T1hat.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    BOOST_CHECK_CLOSE(T0hat.t().x(), T0.t().x(), 1e-4);
    BOOST_CHECK_CLOSE(T0hat.t().y(), T0.t().y(), 1e-4);
    BOOST_CHECK_CLOSE(T0hat.t().z(), T0.t().z(), 1e-4);
    BOOST_CHECK_CLOSE(T0hat.q().w(), T0.q().w(), 1e-4);
    BOOST_CHECK_CLOSE(T0hat.q().x(), T0.q().x(), 1e-4);
    BOOST_CHECK_CLOSE(T0hat.q().y(), T0.q().y(), 1e-4);
    BOOST_CHECK_CLOSE(T0hat.q().z(), T0.q().z(), 1e-4);

    BOOST_CHECK_CLOSE(T1hat.t().x(), T1.t().x(), 1e-4);
    BOOST_CHECK_CLOSE(T1hat.t().y(), T1.t().y(), 1e-4);
    BOOST_CHECK_CLOSE(T1hat.t().z(), T1.t().z(), 1e-4);
    BOOST_CHECK_CLOSE(T1hat.q().w(), T1.q().w(), 1e-4);
    BOOST_CHECK_CLOSE(T1hat.q().x(), T1.q().x(), 1e-4);
    BOOST_CHECK_CLOSE(T1hat.q().y(), T1.q().y(), 1e-4);
    BOOST_CHECK_CLOSE(T1hat.q().z(), T1.q().z(), 1e-4);
}

BOOST_AUTO_TEST_CASE(TestTimeSyncAttFactorProblem)
{
    srand(444444);
    Matrix3d Q = Matrix3d::Identity();
    SO3d qref = SO3d::random();
    Vector3d w(0.5,1.0,-2.0);
    double dt_true = 0.2;
    double dt_hat = 0.0;
    SO3d q = qref + (-dt_true * w);

    ceres::Problem problem;
    problem.AddParameterBlock(&dt_hat, 1);
    problem.AddResidualBlock(TimeSyncAttFactor::Create(qref.array(), q.array(),
                                                       w, Q),
                             nullptr,
                             &dt_hat);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    BOOST_CHECK_CLOSE(dt_true, dt_hat, 1e-4);
}

BOOST_AUTO_TEST_CASE(TestSO3OffsetFactorProblem)
{
    srand(444444);
    Matrix3d Q = Matrix3d::Identity();
    SO3d q = SO3d::random();
    SO3d q_off = SO3d::random();
    SO3d q_ref = q * q_off;
    SO3d q_off_hat = SO3d::identity();

    ceres::Problem problem;
    problem.AddParameterBlock(q_off_hat.data(), 4, SO3Parameterization::Create());
    problem.AddResidualBlock(SO3OffsetFactor::Create(q_ref.array(), q.array(), Q),
                             nullptr,
                             q_off_hat.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    BOOST_CHECK_CLOSE(q_off.w(), q_off_hat.w(), 1e-4);
    BOOST_CHECK_CLOSE(q_off.x(), q_off_hat.x(), 1e-4);
    BOOST_CHECK_CLOSE(q_off.y(), q_off_hat.y(), 1e-4);
    BOOST_CHECK_CLOSE(q_off.z(), q_off_hat.z(), 1e-4);
}

BOOST_AUTO_TEST_CASE(TestSE3OffsetFactorProblem)
{
    srand(444444);
    Matrix<double,6,6> Q = Matrix<double,6,6>::Identity();
    SE3d T = SE3d::random();
    SE3d T_off = SE3d::random();
    SE3d T_ref = T * T_off;
    SE3d T_off_hat = SE3d::identity();

    ceres::Problem problem;
    problem.AddParameterBlock(T_off_hat.data(), 7, SE3Parameterization::Create());
    problem.AddResidualBlock(SE3OffsetFactor::Create(T_ref.array(), T.array(), Q),
                             nullptr,
                             T_off_hat.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.num_threads = 4;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    BOOST_CHECK_CLOSE(T_off.t().x(), T_off_hat.t().x(), 1e-4);
    BOOST_CHECK_CLOSE(T_off.t().y(), T_off_hat.t().y(), 1e-4);
    BOOST_CHECK_CLOSE(T_off.t().z(), T_off_hat.t().z(), 1e-4);
    BOOST_CHECK_CLOSE(T_off.q().w(), T_off_hat.q().w(), 1e-4);
    BOOST_CHECK_CLOSE(T_off.q().x(), T_off_hat.q().x(), 1e-4);
    BOOST_CHECK_CLOSE(T_off.q().y(), T_off_hat.q().y(), 1e-4);
    BOOST_CHECK_CLOSE(T_off.q().z(), T_off_hat.q().z(), 1e-4);
}

BOOST_AUTO_TEST_SUITE_END()
