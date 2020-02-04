#include <gtest/gtest.h>
#include <codi.hpp>
#include <Eigen/Dense>
#include "../src/sde.h"
#include "../src/RungeKutta.hpp"
#include "../src/StochasticLift.hpp"
#include "../src/Pricer.hpp"
#include "../src/VectorFieldSabr.hpp"
#include "../src/VectorFieldHeston.hpp"

class StochasticLiftTest : public ::testing::Test {
public:
//    void SetUp() override {
//        double a = 1.0;
//        double b = 0.4;
//        double beta = 0.9;
//        double rho = -0.7;
//        const sde::Sabr<double> vecField(a, b, beta, rho);
//        const sde::Sabr<codi::RealReverse> vecFieldDiff(a, b, beta, rho);
//        const sde::RungeKutta2<double> rk;
//        const sde::RungeKutta2<codi::RealReverse> rkDiff;
//        int numStepRk = 1;
//
//        _ini << 1., 0.3;
//        _lift = std::make_unique<sde::StochasticLift<sde::RungeKutta2<double>, sde::RungeKutta2<codi::RealReverse>, 2>>(
//            rk, rkDiff, vecField, vecFieldDiff, numStepRk);
//        _rk = std::make_shared<sde::RungeKutta2<double>>(rk);
//        _sabr = std::make_unique<sde::Sabr<double>>(a, b, beta, rho);
//        _bm << 0.1, 0.1;
//
//        double mu = 0.05;
//        double kappa = 2.0;
//        double theta = 0.09;
//        double xi = 0.1;
//        const sde::Heston<double> heston(mu, kappa, theta, xi, rho);
//        const sde::Heston<codi::RealReverse> hestonDiff(mu, kappa, theta, xi, rho);
//        _liftHeston = std::make_unique<sde::StochasticLift<sde::RungeKutta2<double>, sde::RungeKutta2<codi::RealReverse>, 2>>(
//            rk, rkDiff, heston, hestonDiff, numStepRk);
// 
//    }
//    std::unique_ptr<sde::StochasticLift<sde::RungeKutta2<double>, sde::RungeKutta2<codi::RealReverse>, 2>> _lift;
//    std::unique_ptr<sde::StochasticLift<sde::RungeKutta2<double>, sde::RungeKutta2<codi::RealReverse>, 2>> _liftHeston;
// 
//    std::shared_ptr<sde::RungeKutta2<double>> _rk;
//    std::unique_ptr<sde::Sabr<double>> _sabr;
//    sde::vector_type<double, 2> _bm;
//    sde::vector_type<double, 2> _ini;
};

//TEST_F(StochasticLiftTest, getLiftedIni) {
//    sde::vector_type<double, 2> ini;
//    ini << 2.0, 3.0;
//    auto v = _sabr->calcV(ini);
//    sde::vector_type<codi::RealReverse, 6> expected;
//    expected << 2.0, 3.0, v(0,0), v(0,1), v(1,0), v(1,1);
//    auto actual = _lift->getLiftedIni<codi::RealReverse>(ini);
//    for (int i = 0; i < 6; ++i) {
//        EXPECT_EQ(expected(i), actual(i));
//    }
//}
//
//TEST_F(StochasticLiftTest, evolveJacobiInv) {
//    //calculate actual
//    auto actual = _lift->evolveJacobiInv(_ini, _bm);
//
//    //calculate expected differential by finite difference method
//    auto liftedIni = _lift->getLiftedIni<double>(_ini);
//    sde::vector_type<double, 6> liftedIniPlus = liftedIni;
//    sde::vector_type<double, 6> liftedIniMinus = liftedIni;
//
//    Eigen::Matrix2d jac;
//    for (int j = 0; j < 2; ++j) {
//        liftedIniPlus(j) += 0.0001;
//        auto expectedPlus = _rk->solveIterative(1.0, 1, _sabr->getLiftedV(_bm), liftedIniPlus);
//        liftedIniMinus(j) -= 0.0001;
//        auto expectedMinus= _rk->solveIterative(1.0, 1, _sabr->getLiftedV(_bm), liftedIniMinus);
// 
//        for (int i = 0; i < 2; ++i) {
//            jac(i, j) = (expectedPlus(i) - expectedMinus(i))/0.0002;
//        }
//        liftedIniPlus = liftedIni;
//        liftedIniMinus = liftedIni;
//    }
//    auto expected = jac.inverse();
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 2; ++j) {
//
//            std::cout << "evolveJacobi = " << i << ", " << j << " " << expected(i, j) << ", " << actual(i, j) << std::endl;
//            EXPECT_NEAR(expected(i, j), actual(i,j), 1e-6);
//        }
//    }
//    std::cout << expected << std::endl;
//    std::cout << actual<< std::endl;
//}
//
//TEST_F(StochasticLiftTest, evolveXi) 
//{
//    auto actual = _lift->evolveXi(_ini, _bm);
//    std::cout << "xi = " << actual << std::endl;
//}
//TEST_F(StochasticLiftTest, XiPrice) 
//{
//    const double maturity = 1;
//    std::size_t numSteps = 2;
//    std::size_t pathNum = 10;
//    const double strike = 1.05;
//    
//    sde::MtNormal generator(100000);
//    int Size = 2;
//    sde::vector_type<double, 2> ini = _ini;
// 
//    const double dt = maturity / numSteps;
//    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> path;
//    path.resize(2, pathNum);
//    for (int p = 0; p < pathNum; ++p) {
//        const Eigen::MatrixXd& normal = generator.get(Size, numSteps);
//        const Eigen::MatrixXd& bm = sqrt(dt) * normal;
//
//        auto x = ini;
//        for (std::size_t i = 0; i < numSteps; ++i) {
//            x = _lift->evolveXi(x, bm.col(i));
//        }
//        path.col(p) = x;
//    }
//    std::cout << "xiPrice = " << path << std::endl;
//}
//TEST_F(StochasticLiftTest, evolveZeta) 
//{
//    //calculate actual
//    auto actual = _lift->evolveZeta(_ini, _bm);
//    std::cout << actual << std::endl;
//}
//
//
//TEST_F(StochasticLiftTest, evolveX)
//{
//    //calculate actual
//    auto actual = _lift->evolve(_ini, _bm);
//    std::cout << actual << std::endl;
//}
//
//TEST_F(StochasticLiftTest, priceSabr) {
//    const double maturity = 1;
//    std::size_t numSteps = 2;
//    std::size_t pathNum = 100;
//    const double strike = 1.05;
//    
//    sde::MtNormal generator(100000);
//    auto path = _lift->generatePath(maturity, numSteps, pathNum, generator, _ini);
//    
//    std::cout << "path" << std::endl;
//    std::cout << path << std::endl;
//
//    std::cout << "und" << std::endl;
//    auto und = path.row(0);
//    std::cout << und << std::endl;
//
//    std::cout << "und - strike" << std::endl;
//    auto pay = und.array() - strike; 
//    std::cout << pay << std::endl;
//    
//    std::cout << "payoff" << std::endl;
//    auto payoff = pay.max(0.0);
//    std::cout << payoff << std::endl;
//
//    std::cout << "price" << std::endl;
//    std::cout << payoff.isNaN().select(0, payoff).mean() << std::endl;
//    std::cout << "price by pricer" << std::endl;
//
//    double price = sde::Pricer::callPrice(path, strike);
//    std::cout << "price Sabr = " << price << std::endl;
//    double forward = sde::Pricer::forwardPrice(path, 1.0);
//    std::cout << forward << std::endl;
//}

////TEST_F(StochasticLiftTest, priceHeston) {
////    const double maturity = 1;
////    std::size_t numSteps = 2;
////    std::size_t pathNum = 1000;
////    const double strike = 1.05;
////    
////    sde::MtNormal generator(100000);
////    auto path = _liftHeston->generatePath(maturity, numSteps, pathNum, generator, _ini);
////    
////    std::cout << "path" << std::endl;
////    std::cout << path << std::endl;
////
////    std::cout << "und" << std::endl;
////    auto und = path.row(0);
////    std::cout << und << std::endl;
////
////    std::cout << "und - strike" << std::endl;
////    auto pay = und.array() - strike - 0.0401194;
////    std::cout << pay << std::endl;
////    
////    std::cout << "payoff" << std::endl;
////    auto payoff = pay.max(0.0);
////    std::cout << payoff << std::endl;
////
////    std::cout << "price" << std::endl;
////    std::cout << payoff.isNaN().select(0, payoff).mean() << std::endl;
////    std::cout << "price by pricer" << std::endl;
////
////    double price = sde::Pricer::callPrice(path, strike);
////    std::cout << "price Heston = " << price << std::endl;
////    double forward = sde::Pricer::forwardPrice(path, 1.0);
////    std::cout << forward << std::endl;
////}