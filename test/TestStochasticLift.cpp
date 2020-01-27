#include <gtest/gtest.h>
#include <codi.hpp>
#include "../src/sde.h"
#include "../src/RungeKutta.hpp"
#include "../src/StochasticLift.hpp"

class StochasticLiftTest : public ::testing::Test {
public:
    void SetUp() override {
        double stepsize = 0.5;
        double a = 1;
        double b = 0.3;
        double beta = 0.9;
        double rho = -0.7;
        const sde::Sabr<double> vecField(a, b, beta, rho);
        const sde::Sabr<codi::RealReverse> vecFieldDiff(
            static_cast<codi::RealReverse>(a), 
            static_cast<codi::RealReverse>(b), 
            static_cast<codi::RealReverse>(beta), 
            static_cast<codi::RealReverse>(rho));
        const sde::RungeKutta5<double> rk;
        const sde::RungeKutta5<codi::RealReverse> rkDiff;
        sde::vector_type<double, 2> ini;
        ini << 100., 0.3;
        _lift = std::make_unique<sde::StochasticLift<sde::RungeKutta5<double>, sde::RungeKutta5<codi::RealReverse>, 2>>(
            stepsize, rk, rkDiff, vecField, vecFieldDiff, ini);
    }
    std::unique_ptr<sde::StochasticLift<sde::RungeKutta5<double>, sde::RungeKutta5<codi::RealReverse>, 2>> _lift;
 
};

TEST_F(StochasticLiftTest, conructor) {
}