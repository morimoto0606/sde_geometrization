#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Core>
#include "../src/sde.h"
#include "../src/VectorField.hpp"

class VecFieldTest: public ::testing::Test {
public:
    void SetUp() override {
        _a = 1;
        _b = 0.2;
        _beta = -.9;
        _rho = -0.7;

        _sabr = std::make_unique<sde::Sabr<double>>(_a, _b, _beta, _rho);
    }
public:
    double _a;
    double _b;
    double _beta;
    double _rho;
    std::unique_ptr<sde::Sabr<double>> _sabr;
};


TEST_F(VecFieldTest, getV0) {
    auto f = _sabr->getV0();
}

TEST_F(VecFieldTest, getLiftedV) {
    sde::Sabr<double>::vector_type bm;
    bm << -0.32, 0.23;
    auto vecFields = _sabr->getLiftedV(bm);
}