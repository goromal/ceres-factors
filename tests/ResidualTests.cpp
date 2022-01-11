#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <chrono>
#include <SO3.h>
#include <SE3.h>
#include <ceres/ceres.h>
#include "Factors.h"
#include "tests/SO3ComponentFactors.h"
#include "Parameterizations.h"

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

    for(auto r : res) BOOST_CHECK_CLOSE(r, 0.0, 1e-8); 
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
    
    auto r = res2Eigen(res);
    std::cout << "r = " << r.transpose() << std::endl;
    // TODO
}

// BOOST_AUTO_TEST_CASE(TestSO3OMinusFactorRes)
// {
//     srand(444444);
//     double q_hat[4];
//     q_hat[0] = 1.0;
//     q_hat[1] = 0.0;
//     q_hat[2] = 0.0;
//     q_hat[3] = 0.0;
//     SO3d q = SO3d::random();
//     auto q_diff = SO3d::identity() - q;

//     ceres::Problem problem;
//     problem.AddResidualBlock(SO3OMinusFactor::Create(q.array()), nullptr, q_hat);
    
//     std::vector<double> res;
//     ceres::CRSMatrix jac;
//     problem.Evaluate(ceres::Problem::EvaluateOptions(), nullptr, &res, nullptr, &jac);
//     auto r = res2Eigen(res);

//     for (unsigned int i = 0; i < res.size(); i++) BOOST_CHECK_CLOSE(r(i,0), q_diff(i,0), 1e-8);
// }

// ===================================================================================================
// bool Problem::Evaluate(const Problem::EvaluateOptions &options, double *cost, 
//   vector<double> *residuals, vector<double> *gradient, CRSMatrix *jacobian)

// BOOST_AUTO_TEST_CASE(TestSO3FactorResiduals)
// {
//     srand(444444);
//     Matrix3d Q = Matrix3d::Identity();
//     SO3d q = SO3d::random();
//     SO3d qhat = SO3d::identity();
//     Vector3d qhat_minus_q = qhat - q;
    
//     ceres::Problem problem;
//     problem.AddParameterBlock(qhat.data(), 4, SO3Parameterization::Create());

//     ceres::ResidualBlockId res_id = 
//         problem.AddResidualBlock(SO3Factor::Create(q.array(), Q), 
//                                  nullptr,
//                                  qhat.data());

//     double cost;
//     double res[3];
//     problem.EvaluateResidualBlock(res_id, false, &cost, res, nullptr);

//     std::cout << "0: " << res[0] << std::endl;
//     std::cout << "1: " << res[1] << std::endl;
//     std::cout << "2: " << res[2] << std::endl;

//     BOOST_CHECK_CLOSE(res[0], qhat_minus_q.x(), 1e-8);
//     BOOST_CHECK_CLOSE(res[1], qhat_minus_q.y(), 1e-8);
//     BOOST_CHECK_CLOSE(res[2], qhat_minus_q.z(), 1e-8);
// }

// BOOST_AUTO_TEST_CASE(TestSO3FactorResidualsAndJacobians)
// {
//     srand(444444);
//     Matrix3d Q = Matrix3d::Identity();
//     SO3d q = SO3d::random();
//     std::cout << "q = " << q << std::endl;
//     SO3d qhat = SO3d::identity();
//     Vector3d minus = qhat - q;
//     std::cout << "qhat - q = " << (minus).transpose() << std::endl;
//     // Vector3d qhat_minus_q = qhat - q;
//     // std::cout << "qhat_minus_q = " << qhat_minus_q.transpose() << std::endl;
    
//     ceres::Problem problem;
//     problem.AddParameterBlock(qhat.data(), 4, SO3Parameterization::Create());

//     std::cout << "qhat - q = " << (minus).transpose() << std::endl;

//     // std::cout << "qhat_minus_q = " << qhat_minus_q.transpose() << std::endl;

//     ceres::ResidualBlockId res_id = 
//         problem.AddResidualBlock(SO3Factor::Create(q.array(), Q), 
//                                  nullptr,
//                                  qhat.data());

//     std::cout << "qhat - q = " << (minus).transpose() << std::endl;

//     // std::cout << "qhat_minus_q = " << qhat_minus_q.transpose() << std::endl;

//     double cost;
//     double res[3];
//     double jac1[3], jac2[3], jac3[3];
//     double* jac[3] = {jac1, jac2, jac3};
//     problem.EvaluateResidualBlock(res_id, false, &cost, res, jac);

//     // std::cout << "qhat_minus_q = " << qhat_minus_q.transpose() << std::endl;
//     // Vector3d minus = qhat - q;
//     std::cout << "qhat - q = " << (minus).transpose() << std::endl;
//     std::cout << "0: " << res[0] << std::endl;
//     std::cout << "1: " << res[1] << std::endl;
//     std::cout << "2: " << res[2] << std::endl;

//     // BOOST_CHECK_CLOSE(res[0], qhat_minus_q.x(), 1e-8);
//     // BOOST_CHECK_CLOSE(res[1], qhat_minus_q.y(), 1e-8);
//     // BOOST_CHECK_CLOSE(res[2], qhat_minus_q.z(), 1e-8);

//     BOOST_CHECK_CLOSE(res[0], (minus).x(), 1e-8);
//     BOOST_CHECK_CLOSE(res[1], (minus).y(), 1e-8);
//     BOOST_CHECK_CLOSE(res[2], (minus).z(), 1e-8);

//     Matrix3d true_jacobian = Matrix3d::Identity();

//     for (unsigned int i = 0; i < 3; ++i) 
//         for (unsigned int j = 0; j < 3; ++j)
//         {
//             std::cout << i << ", " << j << ": " << jac[i][j] << std::endl;
//             BOOST_CHECK_CLOSE(jac[i][j], true_jacobian(i,j), 1e-8);
//         }
// }

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
