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
        const sde::vector_type<double, Size>& bm) const = 0;
 
    sde::vector_type<double, Size> generateOnePath(
        std::size_t numSteps,
        const Eigen::MatrixXd& bm,
        const sde::vector_type<double, Size>& ini) const
    {
        sde::vector_type<double, Size> x = ini;
        for (std::size_t i = 0; i < numSteps; ++i) {
            x = this->evolve(x, bm.col(i));
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
        const double dt = maturity / numSteps;
        Eigen::MatrixXd path(Size, pathNum);
        for (int p = 0; p < pathNum; ++p) {
            const Eigen::MatrixXd& normal = generator.get(Size, numSteps);
            const Eigen::MatrixXd& bm = sqrt(dt) * normal;
            const sde::vector_type<double, Size>& x = this->generateOnePath(numSteps, bm, ini);
            path.col(p) = x;
        }
        return path;
    }
 
};

}