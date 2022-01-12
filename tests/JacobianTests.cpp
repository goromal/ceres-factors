#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <chrono>
#include <SO3.h>
#include <SE3.h>
#include "ceres-factors/Parameterizations.h"
#include "ceres-factors/Factors.h"
#include "ceres-factors/tests/SO3ComponentFactors.h"
#include <ceres/ceres.h>

using namespace Eigen;

BOOST_AUTO_TEST_SUITE(TestAutoDiff)

MatrixXd CRS2Eigen(const ceres::CRSMatrix &J)
{
    MatrixXd JEig;
    JEig.resize(J.num_rows, J.num_cols);
    JEig.setZero();
    unsigned int j_base = 0;
    unsigned int val_cnt = 0;
    for (unsigned int i = 0; i < J.num_rows; i++) {
        for (unsigned int j = j_base; j < J.rows[i+1]; j++) {
            JEig(i, J.cols[j]) = J.values[val_cnt];
            val_cnt++;
        }
        j_base = J.rows[i+1];
    }
    return JEig;
}

BOOST_AUTO_TEST_CASE(TestSO3Jac)
{
    const uint16_t nSO3Jac = 12;
    double SO3Jac[nSO3Jac] = {  0,   0,   0,
                              0.5,   0,   0,
                                0, 0.5,   0,
                                0,   0, 0.5};

    ceres::LocalParameterization* fn = SO3Parameterization::Create();

    double x[4];
    x[0] = 1.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[3] = 0.0;

    double J[12];
    BOOST_CHECK(fn->ComputeJacobian(x, J));

    for (uint16_t i = 0; i < nSO3Jac; i++)
        BOOST_CHECK_CLOSE(J[i], SO3Jac[i], 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3ConstructorFactorJac)
{
    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;

    ceres::Problem problem;
    problem.AddResidualBlock(SO3ConstructorFactor::Create(), nullptr, q_hat);
    
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jac);
    auto J = CRS2Eigen(jac);

    for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = 0; j < 4; j++)
            BOOST_CHECK_CLOSE(J(i,j), (i==j) ? 1.0 : 0.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3ResMapFactorJac)
{
    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;

    ceres::Problem problem;
    problem.AddResidualBlock(SO3ResMapFactor::Create(), nullptr, q_hat);
    
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jac);
    auto J = CRS2Eigen(jac);

    for (unsigned int i = 0; i < 4; i++)
        for (unsigned int j = 0; j < 4; j++)
            BOOST_CHECK_CLOSE(J(i,j), (i==j) ? 1.0 : 0.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3TrivialOMinusFactorJac)
{
    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;

    ceres::Problem problem;
    problem.AddResidualBlock(SO3TrivialOMinusFactor::Create(), nullptr, q_hat);
    
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jac);
    auto J = CRS2Eigen(jac);

    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 4; j++)
            BOOST_CHECK_CLOSE(J(i,j), 0.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3CastFactorJac)
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
    
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jac);
    auto J = CRS2Eigen(jac);

    for (unsigned int i = 0; i < 2; i++)
        for (unsigned int j = 0; j < 4; j++)
            BOOST_CHECK_CLOSE(J(i,j), (j==0) ? 1.0 : 0.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(TestSO3OMinusFactorJac)
{
    Matrix<double, 3, 4> SO3OMinusFactorJac;
    SO3OMinusFactorJac << 0, 1, 0, 0,
                          0, 0, 1, 0,
                          0, 0, 0, 1;

    double q_hat[4];
    q_hat[0] = 1.0;
    q_hat[1] = 0.0;
    q_hat[2] = 0.0;
    q_hat[3] = 0.0;
    SO3d q = SO3d::identity();

    ceres::Problem problem;
    problem.AddResidualBlock(SO3OMinusFactor::Create(q.array()), nullptr, q_hat);
    
    ceres::CRSMatrix jac;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, nullptr, nullptr, &jac);
    auto J = CRS2Eigen(jac);

    for (unsigned int i = 0; i < 3; i++)
        for (unsigned int j = 0; j < 4; j++)
            BOOST_CHECK_CLOSE(J(i,j), SO3OMinusFactorJac(i,j), 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()
