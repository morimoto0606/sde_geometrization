#pragma once

#include "sde.h"
#include "RndNormal.hpp"

namespace sde {
template <std::size_t Size>
class Scheme {
public:
    virtual ~Scheme() = default;

    virtual sde::vector_type<double, Size> evolve(
        const sde::vector_type<double, Size>& prev,
        const sde::vector_type<double, Size>& dB,
        double dt) const = 0;
 
    sde::vector_type<double, Size> generateOnePath(
        std::size_t numSteps,
        const Eigen::MatrixXd& dB,
        double dt,
        const sde::vector_type<double, Size>& ini) const
    {
        sde::vector_type<double, Size> x = ini;
        for (std::size_t i = 0; i < numSteps; ++i) {
            x = this->evolve(x, dB.col(i), dt);
        }
        return x;
    }

    Eigen::MatrixXd generatePath(
        const double maturity,
        const std::size_t numSteps,
        const std::size_t pathNum,
        const RndNormal& generator,
        const sde::vector_type<double, Size>& ini) const 
    {   
        const double dt = maturity / static_cast<double>(numSteps);
        Eigen::MatrixXd path(pathNum, Size);
        const std::vector<Eigen::MatrixXd>& normals = generator.get(pathNum , Size, numSteps);
        for (int p = 0; p < pathNum; ++p) {
            const Eigen::MatrixXd& dB = sqrt(dt) * normals[p];
            const sde::vector_type<double, Size>& x = this->generateOnePath(numSteps, dB, dt, ini);
            path.row(p) = x;
        }
        return path;
    }
 
};

}