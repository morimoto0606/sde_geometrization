#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <Eigen/Core>
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
#include "../src/RungeKutta.hpp"
#include "../src/sde.h"
#include <codi.hpp>

using namespace std;
using namespace Eigen;
using namespace autodiff;


class RungeKuttaTest : public ::testing::Test {
public:
    virtual void SetUp() {
        _a << 0.,0.,0.,
              1.,0.,0.,
              2.,3.,0.;

        _h = 0.1;

    }
    Matrix3d _a;
    double _h;
};

TEST_F(RungeKuttaTest, exp) {
    sde::RungeKutta5<autodiff::var> rk;
    auto vecField = [this](const Eigen::Matrix<autodiff::var, 3, 1>& x){return _a * x;};
    Eigen::VectorXvar x(3);
    x << 3.,2.,1.;   
    auto actual = rk.solve(_h, vecField, x);

    Eigen::Vector3d y;
    y << 3.,2.,1.;   
 
    auto expected = (Eigen::MatrixXd::Identity(3,3)
        + _h * _a + 0.5 * _h * _h * _a * _a) * y;

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(expected(i, 0), static_cast<double>(actual(i, 0)), 1e-4);
    }
    
}

TEST_F(RungeKuttaTest, expDiff) {
    sde::RungeKutta5<autodiff::var> rk;
    auto vecField = [this](const Eigen::Matrix<autodiff::var, 3, 1>& x){return _a * x;};
    Eigen::VectorXvar x(3);
    x << 3.,2.,1.;   

    auto u = rk.solve(_h, vecField, x);

    auto grad1 = autodiff::gradient(u(0,0), x);
    auto grad2 = autodiff::gradient(u(1,0), x);
    auto grad3 = autodiff::gradient(u(2,0), x);

    Eigen::Matrix3d actual;
    actual.row(0) = grad1;
    actual.row(1) = grad2;
    actual.row(2) = grad3;
    

    Eigen::Vector3d y;
    y << 3.,2.,1.;   
    
    Eigen::Vector3d y1p;
    y1p << 3.+0.001,2.,1.;   
    Eigen::Vector3d y1m;
    y1m << 3.-0.001,2.,1.;   

    Eigen::Vector3d y2p;
    y2p << 3.,2.+0.001,1.;   
    Eigen::Vector3d y2m;
    y2m << 3.,2.-0.001,1.;   

    Eigen::Vector3d y3p;
    y3p << 3.,2.,1.+0.001;   
    Eigen::Vector3d y3m;
    y3m << 3.,2.,1.-0.001;   


    auto x1p = (Eigen::MatrixXd::Identity(3,3)
        + _h * _a + 0.5 * _h * _h * _a * _a) * y1p;

    auto x1m = (Eigen::MatrixXd::Identity(3,3)
        + _h * _a + 0.5 * _h * _h * _a * _a) * y1m;

    auto x2p = (Eigen::MatrixXd::Identity(3,3)
        + _h * _a + 0.5 * _h * _h * _a * _a) * y2p;

    auto x2m = (Eigen::MatrixXd::Identity(3,3)
        + _h * _a + 0.5 * _h * _h * _a * _a) * y2m;

    auto x3p = (Eigen::MatrixXd::Identity(3,3)
        + _h * _a + 0.5 * _h * _h * _a * _a) * y3p;

    auto x3m = (Eigen::MatrixXd::Identity(3,3)
        + _h * _a + 0.5 * _h * _h * _a * _a) * y3m;

    auto expected1 = (x1p - x1m) / 0.002;
    auto expected2 = (x2p - x2m) / 0.002;
    auto expected3 = (x3p - x3m) / 0.002;

    Eigen::Matrix3d expected;
    expected.col(0) = expected1;
    expected.col(1) = expected2;
    expected.col(2) = expected3;



    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(expected(i, j), actual(i, j), 1e-4);
        }
    }
    
}

TEST_F(RungeKuttaTest, expDiffIterative) {
    sde::RungeKutta5<codi::RealReverse> rk;

    Eigen::Matrix<codi::RealReverse, 3, 3> a;
    a << 0.,0.,0.,
         1.,0.,0.,
         2.,3.,0.;

    int iterNum = 10;
    const codi::RealReverse h = 1.0/iterNum;
    auto vecField = [this, &a](const sde::vector_type<codi::RealReverse, 3>& x){return a * x;};
    auto vecFieldPtr = std::make_shared<sde::function_type<sde::vector_type<codi::RealReverse, 3>>>(vecField);
 
    std::vector<sde::func_ptr_type<sde::vector_type<codi::RealReverse, 3>>> vecFields;
    for (int i =0; i < iterNum; ++i) {
        vecFields.emplace_back(vecFieldPtr);
    }

    sde::vector_type<codi::RealReverse, 3> x;
    x << 3.,2.,1.;   

    codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
    tape.setActive();
    for(size_t i = 0; i < 3; ++i) {
      tape.registerInput(x(i));
    }
    auto z = rk.solveIterative(h, vecFields, x);
 
    tape.registerOutput(z(0));
    tape.registerOutput(z(1));
    tape.registerOutput(z(2));
 
    tape.setPassive();
    std::cout << "f(1 .. 5) = (" << z(0,0) << ", " << z(1,0) << "," << z(2,0) << ")" << std::endl;
    z(0,0).setGradient(1.0);
    tape.evaluate();
    Eigen::Matrix3d df = Eigen::MatrixXd::Zero(3,3);

    std::cout << "df_1/dx(1 .. 5) = (";
    for(size_t i = 0; i < 3; ++i) {
      if(0 != i) {
        std::cout << ", ";
      }
      std::cout << x(i).getGradient();
      df(0, i) = x(i).getGradient();
    }
    std::cout << ")" << std::endl;
    tape.clearAdjoints();

    z(1).setGradient(1.0);
    tape.evaluate();
    std::cout << "df_2/dx(1 .. 5) = (";
    for(size_t i = 0; i < 3; ++i) {
      if(0 != i) {
        std::cout << ", ";
      }
      std::cout << x(i).getGradient();
      df(1, i) = x(i).getGradient();
    }
    std::cout << ")" << std::endl;
    tape.clearAdjoints();

    z(2).setGradient(1.0);
    tape.evaluate();
    std::cout << "df_3/dx(1 .. 5) = (";
    for(size_t i = 0; i < 3; ++i) {
      if(0 != i) {
        std::cout << ", ";
      }
      std::cout << x(i).getGradient();
      df(2, i) = x(i).getGradient();
    }
    std::cout << ")" << std::endl;
    tape.clearAdjoints();

 
    Eigen::Vector3d y;
    y << 3.,2.,1.;   
    
    Eigen::Vector3d y1p;
    y1p << 3.+0.001,2.,1.;   
    Eigen::Vector3d y1m;
    y1m << 3.-0.001,2.,1.;   

    Eigen::Vector3d y2p;
    y2p << 3.,2.+0.001,1.;   
    Eigen::Vector3d y2m;
    y2m << 3.,2.-0.001,1.;   

    Eigen::Vector3d y3p;
    y3p << 3.,2.,1.+0.001;   
    Eigen::Vector3d y3m;
    y3m << 3.,2.,1.-0.001;   

    double t = 1;
    auto x1p = (Eigen::MatrixXd::Identity(3,3)
        + t * _a + 0.5 * t * t * _a * _a) * y1p;

    auto x1m = (Eigen::MatrixXd::Identity(3,3)
        + t * _a + 0.5 * t * t * _a * _a) * y1m;

    auto x2p = (Eigen::MatrixXd::Identity(3,3)
        + t * _a + 0.5 * t * t * _a * _a) * y2p;

    auto x2m = (Eigen::MatrixXd::Identity(3,3)
        + t * _a + 0.5 * t * t * _a * _a) * y2m;

    auto x3p = (Eigen::MatrixXd::Identity(3,3)
        + t * _a + 0.5 * t * t * _a * _a) * y3p;

    auto x3m = (Eigen::MatrixXd::Identity(3,3)
        + t * _a + 0.5 * t * t * _a * _a) * y3m;

    auto expected1 = (x1p - x1m) / 0.002;
    auto expected2 = (x2p - x2m) / 0.002;
    auto expected3 = (x3p - x3m) / 0.002;

    Eigen::Matrix3d expected;
    expected.col(0) = expected1;
    expected.col(1) = expected2;
    expected.col(2) = expected3;

    for (int i = 0; i < 3; ++i) {
        std::cout << "expexted " << i << " = " << expected.col(i) << std::endl;
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(expected(i, j), df(i, j), 1e-4);
        }
    }
    
}