#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Core>
#include <codi.hpp>
#include "../src/sde.h"
#include "../src/VectorField.hpp"
#include "../src/VectorFieldSabr.hpp"

class VecFieldTest: public ::testing::Test {
public:
    void SetUp() override {
        _a = 1;
        _b = 0.4;
        _beta = 0.9;
        _rho = -0.7;
        _sabr = std::make_unique<sde::Sabr>(_a, _b, _beta, _rho);
        _sabrDiff = std::make_unique<sde::Sabr>(_a,_b,_beta,_rho);
    }
public:
    double _a;
    double _b;
    double _beta;
    double _rho;
    std::unique_ptr<sde::Sabr> _sabr;
    std::unique_ptr<sde::Sabr> _sabrDiff;
};


TEST_F(VecFieldTest, getV0) {
    auto f = _sabr->getV0<double>();
}
TEST_F(VecFieldTest, getV0Diff) {
    auto f = _sabrDiff->getV0<codi::RealReverse>();
}

TEST_F(VecFieldTest, getLiftedV) {
    sde::vector_type<double, 2> bm;
    bm << -0.32, 0.23;
    auto vecFields = _sabr->getLiftedV<double>(bm);
    sde::lifted_type<double, 2> x;
    x << 1.0, 0.3, 1, 0, 0, 1;
    sde::lifted_type<double, 2> y0 = (*vecFields[0])(x);
    sde::lifted_type<double, 2> y1 = (*vecFields[1])(x);

    std::cout << "getLiftedV 0 = " << y0 << std::endl;
    std::cout << "getLiftedV 1 = " << y1 << std::endl;

}


TEST_F(VecFieldTest, getLiftedVDiff) {
    sde::vector_type<double, 2> bm;
    bm << -0.32, 0.23;
    auto vecFields = _sabrDiff->getLiftedV<codi::RealReverse>(bm);
}