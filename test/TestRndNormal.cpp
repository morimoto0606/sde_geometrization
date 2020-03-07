#include <gtest/gtest.h>
#include "../src/RndNormal.hpp"

class RndNormalTest : public ::testing::Test {
public:
    void SetUp() override {
        _seed = 2;
        _dim = 2;
        _size = 1000;
    }

    std::size_t _seed;
    std::size_t _size;
    std::size_t _dim;
};


TEST_F(RndNormalTest, MtNormal) {
    sde::MtNormal generator(_seed);
    const auto rndNormal = generator.get(1, _dim, _size);
    EXPECT_NEAR(0.0, rndNormal[0].mean(), 0.05);
    std::cout << rndNormal[0].mean() << std::endl;
}