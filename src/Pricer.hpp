#pragma once
#include <Eigen/Dense>
namespace sde {
class Pricer {
public:
    static double callPrice(
        const Eigen::MatrixXd& path,
        double strike) 
    {
        auto und = path.col(0);
        auto pay = und.array() - strike;
        auto payoff = pay.max(0.0);
        double mean = payoff.isNaN().select(0, payoff).mean();
        return mean;
    }

    static std::vector<double> forwardPrice(
        const Eigen::MatrixXd& path) 
    {
        auto und = path.col(0);
        auto payoff = und.array();
        double mean = payoff.isNaN().select(0, payoff).mean();

        auto vol = path.col(1);
        auto payoff1 = vol.array();
        double volmean = payoff1.isNaN().select(0, payoff1).mean();
 
        return std::vector<double>{mean, volmean};
    }
};
}