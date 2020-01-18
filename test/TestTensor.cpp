#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Core>
#include "../src/sde.h"

class TensorTest: public ::testing::Test {

};

TEST_F(TensorTest, constructor) {
    sde::Tensor<double, 2> hoge;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                EXPECT_EQ(0, hoge(i, j, k));
            }
        }
    }
}

TEST_F(TensorTest, accessor) {
    sde::Tensor<double, 2> hoge;
    double d = 3.545;
    hoge(0,1,0) = d;
    EXPECT_EQ(d, hoge(0,1,0));
}