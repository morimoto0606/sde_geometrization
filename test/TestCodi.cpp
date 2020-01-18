#include <gtest/gtest.h>
#include <codi.hpp>
#include <iostream>
#include <Eigen/Core>

class CodiTest : public ::testing::Test {

};

TEST_F(CodiTest, HelloWorld) {

    codi::RealForward x = 4.0;
    x.setGradient(1.0);

    codi::RealForward y = x * x;

    std::cout << "f(4.0) = " << y << std::endl;
    std::cout << "df/dx(4.0) = " << y.getGradient() << std::endl;

}

TEST_F(CodiTest, Reverse_Jacobi) {
auto func = [](const Eigen::Matrix<codi::RealReverse, 5, 1>& x) {
    Eigen::Matrix<codi::RealReverse, 2, 1> y;
    y << 0.0, 1.0;
    for(size_t i = 0; i < 5; ++i) {
        y(0,0) += x(i, 0);
        y(1, 0) *= x(i, 0);
    }
    return y;
};
  
    Eigen::Matrix<codi::RealReverse, 5, 1> x;
    Eigen::Matrix<codi::RealReverse, 2, 1>  y;
    x << 1.0,2.0,3.0,4.0,5.0;
    codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
    tape.setActive();
    for(size_t i = 0; i < 5; ++i) {
      tape.registerInput(x(i, 0));
    }
    y = func(x);
    tape.registerOutput(y(0,0));
    tape.registerOutput(y(1,0));
    tape.setPassive();
    std::cout << "f(1 .. 5) = (" << y(0,0) << ", " << y(1,0) << ")" << std::endl;
    y(0,0).setGradient(1.0);
    tape.evaluate();
    std::cout << "df_1/dx(1 .. 5) = (";
    for(size_t i = 0; i < 5; ++i) {
      if(0 != i) {
        std::cout << ", ";
      }
      std::cout << x(i,0).getGradient();
    }
    std::cout << ")" << std::endl;
    tape.clearAdjoints();
    y(1,0).setGradient(1.0);
    tape.evaluate();
    std::cout << "df_2/dx(1 .. 5) = (";
    for(size_t i = 0; i < 5; ++i) {
      if(0 != i) {
        std::cout << ", ";
      }
      std::cout << x(i, 0).getGradient();
    }
    std::cout << ")" << std::endl;
}

TEST_F(CodiTest, MatrixMulti) {

    Eigen::Matrix<codi::RealReverse, 2, 2> x;
    Eigen::Matrix<codi::RealReverse, 2, 2> y;
    x << 1.0,2.0,
         3.0,4.0;
    y << 1.0,2.0,
         3.0,4.0;
    auto z = x * y;
}

TEST_F(CodiTest, power) {
    codi::RealReverse x = 3.0;
    codi::RealReverse y;
 
    codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
    tape.setActive();

    tape.registerInput(x);

    y = pow(x, 5.);
    tape.registerOutput(y);
    tape.setPassive();
    
    y.setGradient(1.0);
    tape.evaluate();
    EXPECT_NEAR(5 * pow(3., 4), x.getGradient(), 1e-8);
}