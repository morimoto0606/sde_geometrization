#include <gtest/gtest.h>
#include <codi.hpp>
#include <Eigen/Dense>
#include "../src/sde.h"
#include "../src/RungeKutta.hpp"
#include "../src/StochasticLift.hpp"

class StochasticLiftTest : public ::testing::Test {
public:
    void SetUp() override {
        double stepsize = .2;
        double a = 1.0;
        double b = 0.4;
        double beta = 0.9;
        double rho = -0.7;
        const sde::Sabr<double> vecField(a, b, beta, rho);
        const sde::Sabr<codi::RealReverse> vecFieldDiff(a, b, beta, rho);
        const sde::RungeKutta5<double> rk;
        const sde::RungeKutta5<codi::RealReverse> rkDiff;
        _ini << 1., 0.3;
        _lift = std::make_unique<sde::StochasticLift<sde::RungeKutta5<double>, sde::RungeKutta5<codi::RealReverse>, 2>>(
            stepsize, rk, rkDiff, vecField, vecFieldDiff, _ini);
        _rk = std::make_shared<sde::RungeKutta5<double>>(rk);
        _sabr = std::make_unique<sde::Sabr<double>>(a, b, beta, rho);
        _stepSize = stepsize;
        _bm << 1.0, 1.0;
    }
    double _stepSize;
    std::unique_ptr<sde::StochasticLift<sde::RungeKutta5<double>, sde::RungeKutta5<codi::RealReverse>, 2>> _lift;
    std::shared_ptr<sde::RungeKutta5<double>> _rk;
    std::unique_ptr<sde::Sabr<double>> _sabr;
    sde::vector_type<double, 2> _bm;
    sde::vector_type<double, 2> _ini;
};

TEST_F(StochasticLiftTest, getLiftedIni) {
    sde::vector_type<double, 2> ini;
    ini << 2.0, 3.0;
    sde::vector_type<codi::RealReverse, 6> expected;
    expected << 2.0, 3.0, 1.0, 0.0, 0.0, 1.0;
    auto actual = _lift->getLiftedIni<codi::RealReverse>(ini);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(expected(i), actual(i));
    }
}

TEST_F(StochasticLiftTest, evolveJacobiInv) {
    //calculate actual
    auto actual = _lift->evolveJacobiInv(_ini, _bm);

    //calculate expected differential by finite difference method
    sde::vector_type<double, 6> liftedIni;
    liftedIni << 1.0, 0.3, 1., 0., 0., 1.;
    sde::vector_type<double, 6> liftedIniPlus = liftedIni;
    sde::vector_type<double, 6> liftedIniMinus = liftedIni;

    Eigen::Matrix2d jac;
    for (int j = 0; j < 2; ++j) {
        liftedIniPlus(j) += 0.0001;
        auto expectedPlus = _rk->solveIterative(1.0, _sabr->getLiftedV(_bm), liftedIniPlus);
        liftedIniMinus(j) -= 0.0001;
        auto expectedMinus= _rk->solveIterative(1.0, _sabr->getLiftedV(_bm), liftedIniMinus);
        for (int i = 0; i < 2; ++i) {
            jac(i, j) = (expectedPlus(i) - expectedMinus(i))/0.0002;
        }
        liftedIniPlus = liftedIni;
        liftedIniMinus = liftedIni;
    }
    auto expected = jac.inverse();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(expected(i, j), actual(i,j), 1e-6);
        }
    }
    std::cout << expected << std::endl;
    std::cout << actual<< std::endl;
}

TEST_F(StochasticLiftTest, evolveZeta) 
{
    //calculate actual
    auto actual = _lift->evolveZeta(_ini, _bm);
    std::cout << actual << std::endl;
}


TEST_F(StochasticLiftTest, evolveX)
{
    //calculate actual
    auto actual = _lift->evolveX(_ini, _bm);
    std::cout << actual << std::endl;
}

TEST_F(StochasticLiftTest, price) {
    std::size_t pathNum = 100000;
    const double strike = 1.05;
    //auto payoff = [strike](const Eigen::MatrixXd& x){
    //    auto x0 = x.row(0);
    //    return (x0.array() - strike).max(Eigen::ArrayXd::Zero(x0.size())).mean();
    //};
    sde::MtNormal generator(2);
    auto path = _lift->generatePaths(5, pathNum, generator);
    
    std::cout << "path" << std::endl;
    std::cout << path << std::endl;

    std::cout << "und" << std::endl;
    auto und = path.row(0);
    std::cout << und << std::endl;

    std::cout << "und - strike" << std::endl;
    auto pay = und.array() - strike;
    std::cout << pay << std::endl;
    
    std::cout << "payoff" << std::endl;
    auto payoff = pay.max(0.0);
    std::cout << payoff << std::endl;

    std::cout << "price" << std::endl;
    std::cout << payoff.isNaN().select(0, payoff).mean() << std::endl;
 
    //double actual = payoff(path);
    //std::cout << actual << std::endl;

}