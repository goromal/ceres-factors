#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <chrono>
#include <SO3.h>
#include <SE3.h>
#include <ceres/ceres.h>
#include "ceres-factors/Factors.h"
#include "ceres-factors/tests/SO3ComponentFactors.h"
#include "ceres-factors/Parameterizations.h"

using namespace Eigen;

BOOST_AUTO_TEST_SUITE(TestFactors)

MatrixXd res2Eigen(const std::vector<double> res)
{
    MatrixXd resEig;
    resEig.resize(res.size(), 1);
    for (unsigned int i = 0; i < res.size(); i++)
        resEig(i,0) = res[i];
    return resEig;
}

BOOST_AUTO_TEST_CASE(TestSO3ConstructorFactorRes)
{
    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;

    ceres::Problem problem;
    problem.AddResidualBlock(SO3ConstructorFactor::Create(), nullptr, q_hat);

    std::vector<double> res;
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &res, nullptr, &jac);

    for (unsigned int i = 0; i < res.size(); i++) BOOST_CHECK_CLOSE(res[i], q_hat[i], 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3ResMapFactorRes)
{
    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;

    ceres::Problem problem;
    problem.AddResidualBlock(SO3ResMapFactor::Create(), nullptr, q_hat);

    std::vector<double> res;
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &res, nullptr, &jac);

    for (unsigned int i = 0; i < res.size(); i++) BOOST_CHECK_CLOSE(res[i], q_hat[i], 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3TrivialOMinusFactorRes)
{
    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;

    ceres::Problem problem;
    problem.AddResidualBlock(SO3TrivialOMinusFactor::Create(), nullptr, q_hat);
    
    std::vector<double> res;
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &res, nullptr, &jac);

    for (auto r : res) BOOST_CHECK_CLOSE(r, 0.0, 1e-8); 
}

BOOST_AUTO_TEST_CASE(TestSO3CastFactorRes)
{
    srand(444444);
    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;
    SO3d q = SO3d::identity();

    ceres::Problem problem;
    problem.AddResidualBlock(SO3CastFactor::Create(q.array()), nullptr, q_hat);
    
    std::vector<double> res;
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &res, nullptr, &jac);
    
    for (auto r : res) BOOST_CHECK_CLOSE(r, 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3OMinusFactorRes)
{
    srand(444444);
    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;
    SO3d q = SO3d::random();
    auto q_diff = SO3d::identity() - q;

    ceres::Problem problem;
    problem.AddResidualBlock(SO3OMinusFactor::Create(q.array()), nullptr, q_hat);
    
    std::vector<double> res;
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &res, nullptr, &jac);
    auto r = res2Eigen(res);

    for (unsigned int i = 0; i < res.size(); i++) BOOST_CHECK_CLOSE(r(i,0), q_diff(i,0), 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()
