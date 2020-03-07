#include <gtest/gtest.h>
#include "../src/Em.hpp"
#include "../src/Pricer.hpp"

class TestEm : public ::testing::Test {
public:
    void SetUp() override 
    {
        double a = 1.0;
        double b = 0.4;
        double beta = .9;
        double rho = -0.7;
        double x0 = 1.0;
        double x1 = 0.3;
        _em = std::make_unique<sde::SabrEm>(a, b, beta, rho);
        _liftedEm = std::make_unique<sde::LiftedSabrEm>(a, b, beta, rho);
 
        _ini << x0, x1;
        _liftedIni << x0, x1, a*pow(x0, beta)*x1, 0., b*rho*x1, b*sqrt(1.-rho*rho)*x1;
    }
    std::unique_ptr<sde::SabrEm> _em;
    std::unique_ptr<sde::LiftedSabrEm> _liftedEm;
    sde::vector_type<double, 2> _ini;
    sde::vector_type<double, 6> _liftedIni;

};

TEST_F(TestEm, Price) {
    const double maturity = 1.0;
    std::size_t numSteps = 10;
    std::size_t pathNum = 1000;
    const double strike = 1.05;
    
    sde::MtNormal generator(10000);
    auto path = _em->generatePath(maturity, numSteps, pathNum, generator, _ini);
    //std::cout << path << std::endl;
    double price = sde::Pricer::callPrice(path, strike);
    std::cout << price << std::endl;
    auto forward = sde::Pricer::forwardPrice(path);
    std::cout << forward[0] << ", "  << forward[1] << std::endl;
}

TEST_F(TestEm, LiftedPrice) {
    const double maturity = 1.0;
    std::size_t numSteps = 10;
    std::size_t pathNum = 1000;
    const double strike = 1.05;
    
    sde::MtNormal generator(10000);
    auto path = _liftedEm->generatePath(maturity, numSteps, pathNum, generator, _liftedIni);
    //std::cout << path << std::endl;
    double price = sde::Pricer::callPrice(path, strike);
    std::cout << price << std::endl;
    auto forward = sde::Pricer::forwardPrice(path);
    std::cout << forward[0] << ", "  << forward[1] << std::endl;
}