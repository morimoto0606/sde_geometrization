#include <gtest/gtest.h>
#include <cmath>
#include "../src/Pricer.hpp"
#include "../src/sde.h"

class PricerTest : public ::testing::Test {
public:
    void SetUp() override {
        _path.resize(4, 2);
        _path << 2.5, 0.1,
                1.4, 0.2,
                3.0, 0.3, 
                std::nan("1"), std::nan("2");
    }

    Eigen::MatrixXd _path;
};

TEST_F(PricerTest, sabrCall) {

    auto und = _path.col(0);
    sde::vector_type<double, 4> expectedRow;
    expectedRow << 2.5, 1.4, 3.0, std::nan("1");
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(expectedRow(i), und(i));
    }

    const double strike = 2.0;
    auto pay = und.array() - strike;
    sde::vector_type<double, 4> expectedPay;
    expectedPay << 0.5, -0.6, 1.0, std::nan("1");
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(expectedPay(i), pay(i));
    }

    auto payoff = pay.max(0.0);
    sde::vector_type<double, 4> expectedPayoff;
    expectedPayoff << 0.5, 0.0, 1.0, std::nan("1");
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(expectedPayoff(i), payoff(i));
    }

    sde::vector_type<double, 4> payoffMod = payoff.isNaN().select(0, payoff);
    sde::vector_type<double, 4> expectedPayoffMod;
    expectedPayoffMod << 0.5, 0.0, 1.0, 0.0;
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(expectedPayoffMod(i),payoffMod(i));
    }

    double actual = sde::Pricer::callPrice(_path, strike);
    double expected = 1.5 / 4.0;
    EXPECT_DOUBLE_EQ(expected, actual);

}