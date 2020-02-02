#pragma once
#include <Eigen/Dense>
#include "sde.h"
#include "Scheme.hpp"
#include "RndNormal.hpp"


namespace sde {

class SabrEm : public Scheme<2> {
public:
    SabrEm(
        double a,
        double b,
        double beta,
        double rho): 
        _a(a), _b(b), _beta(beta), _rho(rho) 
    {
    }

    sde::vector_type<double, 2> evolve(
        const sde::vector_type<double, 2>& x,
        const sde::vector_type<double, 2>& bm) const override
    {
        sde::vector_type<double, 2> y;
        y(0) = x(0) + _a * x(1) * pow(x(0), _beta) * bm(0);
        y(1) = x(1) + _b * x(1) * (_rho * bm(0) + sqrt(1.- _rho * _rho) * bm(1));
        return y;
    }

private:
    double _a;
    double _b;
    double _beta;
    double _rho;
};

}