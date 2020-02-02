#include <gtest/gtest.h>
#include "../src/Em.hpp"
#include "../src/Pricer.hpp"

class TestEm : public ::testing::Test {
public:
    void SetUp() override 
    {
        double a = 1.0;
        double b = 0.4;
        double beta = 0.9;
        double rho = -0.7;
        _em = std::make_unique<sde::SabrEm>(a, b, beta, rho);
        _ini << 1., 0.3;
    }
    std::unique_ptr<sde::SabrEm> _em;
    sde::vector_type<double, 2> _ini;
};

TEST_F(TestEm, Price) {
    const double maturity = 1.0;
    std::size_t numSteps = 10;
    std::size_t pathNum = 1000;
    const double strike = 1.05;
    
    sde::MtNormal generator(10000);
    auto path = _em->generatePath(maturity, numSteps, pathNum, generator, _ini);
    double price = sde::Pricer::callPrice(path, strike);
    std::cout << price << std::endl;
    double forward = sde::Pricer::forwardPrice(path, 1.0);
    std::cout << forward << std::endl;
}