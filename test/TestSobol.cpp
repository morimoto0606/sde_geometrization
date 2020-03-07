#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Core>
#include <numeric>
#include "../src/sde.h"
#include "../src/sobol.h"
#include <iomanip>

class SobolTest: public ::testing::Test {
};

TEST_F(SobolTest, sobol_points) {
    int N = 10;
    int D = 3;
    auto rnd = sobol_points(N, D);
    EXPECT_DOUBLE_EQ(0.6875, rnd[9][0]);
    EXPECT_DOUBLE_EQ(0.8125, rnd[9][1]);  
    EXPECT_DOUBLE_EQ(0.4375, rnd[9][2]);   
    auto rnd2 = sobol_points(N, D);
    std::vector<double> samples0;
    std::vector<double> samples1;
    for (int i = 0; i < N; ++i) {
        samples0.push_back(rnd2[i][0]);
        samples1.push_back(rnd2[i][1]);
    }
    double ave0 = std::accumulate(samples0.begin(), samples0.end(), 0.) / static_cast<double>(N);
    double ave1 = std::accumulate(samples1.begin(), samples1.end(), 0.) / static_cast<double>(N);

    std::cout <<  ave0 << ", " << ave1 << std::endl;
}

TEST_F(SobolTest, sobol_points_normal) {
    int N = 10000;
    int D = 2;
    auto rnd = sobol_points_normal(N, D, 1);
    std::vector<double> samples0;
    std::vector<double> samples1;
    for (int i = 0; i < N; ++i) {
        samples0.push_back(rnd(i, 0));
        samples1.push_back(rnd(i, 1));
        //for (auto y : x) {
        //    //std::cout << y << " " ;
        //}
        //std::cout << std::endl;
    }
    double ave0 = std::accumulate(samples0.begin(), samples0.end(), 0.) / static_cast<double>(N);
    double ave1 = std::accumulate(samples1.begin(), samples1.end(), 0.) / static_cast<double>(N);

    std::cout <<  ave0 << ", " << ave1 << std::endl;
 
}
