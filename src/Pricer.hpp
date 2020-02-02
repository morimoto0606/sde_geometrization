#pragma once
#include <Eigen/Dense>
namespace sde {
class Pricer {
public:
    static double callPrice(
        const Eigen::MatrixXd& path,
        double strike) 
    {
        auto und = path.row(0);
        auto pay = und.array() - strike;
        auto payoff = pay.max(0.0);
        double mean = payoff.isNaN().select(0, payoff).mean();
        return mean;
    }

    static double forwardPrice(
        const Eigen::MatrixXd& path,
        double strike) 
    {
        auto und = path.row(0);
        auto payoff = und.array() - strike;
        double mean = payoff.isNaN().select(0, payoff).mean();
        return mean;
    }
};
}