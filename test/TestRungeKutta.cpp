#include <gtest/gtest.h>
#include <functional>
#include <iostream>
#include <Eigen/Dense>
#include <autodiff/reverse.hpp>
#include "../src/RungeKutta.hpp"

using namespace std;
using namespace Eigen;
using namespace autodiff;


class RungeKuttaTest : public ::testing::Test {
public:
    virtual void SetUp() {
        _a = 1.0;
        _vecField = [this](double x){return _a * x;};
    }
    double _a;
    std::function<double(double)> _vecField;
};

TEST_F(RungeKuttaTest, Exp) {
    double x = 3.0;
    double expected = _a * x;
    double actual = _vecField(x);
    EXPECT_EQ(expected, actual);
}

TEST_F(RungeKuttaTest, ExpMatrix) {
  Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,3);
  m = (m + Eigen::MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
}

TEST_F(RungeKuttaTest, hoge) {
    var x = 1.0;                                 // the input variable x
    var y = 0.5;                                 // the input variable y
    var z = 2.0;                                 // the input variable z

    var u = x * log(y) * exp(z);                 // the output variable u

    DerivativesX dud = derivativesx(u);          // evaluate all derivatives of u using autodiff::derivativesx!

    var dudx = dud(x);                           // extract the first order derivative du/dx of type var, not double!
    var dudy = dud(y);                           // extract the first order derivative du/dy of type var, not double!
    var dudz = dud(z);                           // extract the first order derivative du/dz of type var, not double!

    DerivativesX d2udxd = derivativesx(dudx);    // evaluate all derivatives of dudx using autodiff::derivativesx!
    DerivativesX d2udyd = derivativesx(dudy);    // evaluate all derivatives of dudy using autodiff::derivativesx!
    DerivativesX d2udzd = derivativesx(dudz);    // evaluate all derivatives of dudz using autodiff::derivativesx!

    var d2udxdx = d2udxd(x);                     // extract the second order derivative d2u/dxdx of type var, not double!
    var d2udxdy = d2udxd(y);                     // extract the second order derivative d2u/dxdy of type var, not double!
    var d2udxdz = d2udxd(z);                     // extract the second order derivative d2u/dxdz of type var, not double!

    var d2udydx = d2udyd(x);                     // extract the second order derivative d2u/dydx of type var, not double!
    var d2udydy = d2udyd(y);                     // extract the second order derivative d2u/dydy of type var, not double!
    var d2udydz = d2udyd(z);                     // extract the second order derivative d2u/dydz of type var, not double!

    var d2udzdx = d2udzd(x);                     // extract the second order derivative d2u/dzdx of type var, not double!
    var d2udzdy = d2udzd(y);                     // extract the second order derivative d2u/dzdy of type var, not double!
    var d2udzdz = d2udzd(z);                     // extract the second order derivative d2u/dzdz of type var, not double!

    cout << "u = " << u << endl;                 // print the evaluated output variable u

    cout << "du/dx = " << dudx << endl;          // print the evaluated first order derivative du/dx
    cout << "du/dy = " << dudy << endl;          // print the evaluated first order derivative du/dy
    cout << "du/dz = " << dudz << endl;          // print the evaluated first order derivative du/dz

    cout << "d2udxdx = " << d2udxdx << endl;     // print the evaluated second order derivative d2u/dxdx
    cout << "d2udxdy = " << d2udxdy << endl;     // print the evaluated second order derivative d2u/dxdy
    cout << "d2udxdz = " << d2udxdz << endl;     // print the evaluated second order derivative d2u/dxdz

    cout << "d2udydx = " << d2udydx << endl;     // print the evaluated second order derivative d2u/dydx
    cout << "d2udydy = " << d2udydy << endl;     // print the evaluated second order derivative d2u/dydy
    cout << "d2udydz = " << d2udydz << endl;     // print the evaluated second order derivative d2u/dydz

    cout << "d2udzdx = " << d2udzdx << endl;     // print the evaluated second order derivative d2u/dzdx
    cout << "d2udzdy = " << d2udzdy << endl;     // print the evaluated second order derivative d2u/dzdy
    cout << "d2udzdz = " << d2udzdz << endl;     // print

}

TEST_F(RungeKuttaTest, constructor) {
    sde::RungeKutta5<double> hoge;
}