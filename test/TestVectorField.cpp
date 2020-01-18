#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Core>
#include "../src/sde.h"
#include "../src/VectorField.hpp"

class VecFieldTest: public ::testing::Test {
    void SetUp() override {
        _a = 1;
        _b = 0.2;
        _beta = -.9;
        _rho = -0.7;

        _sabr = sde::Sabr<double>(_a, _b, _beta, _rho);
    }

    double _a;
    double _b;
    double _beta;
    double _rho;
    sde::Sabr<double> _sabr;
};


TEST_F(VecFieldTest, getV0) {
    auto f = _sabr.getV0();
}