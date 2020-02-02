#pragma once
#include "sde.h"
#include "VectorField.hpp"
namespace sde
{

template <typename T>
class Sabr : public VectorField<T, 2> {
public:
    using vector_type = sde::vector_type<T, 2>;
    using lifted_type = sde::lifted_type<T, 2>;
 
   Sabr(
        const T& a,
        const T& b,
        const T& beta,
        const T& rho)
    : _a(a), _b(b), _beta(beta), _rho(rho)
    {

    }

    std::unique_ptr<VectorField<T, 2>> clone() const override 
    {
        return std::make_unique<Sabr<T>>(*this);
    }


    Eigen::Matrix<T, 2, 2> calcGInv(const sde::vector_type<T, 2>& x) const override
    {
        Eigen::Matrix<T, 2, 2> mat;
        mat(0,0) = pow(_a, 2.) * pow(x(0), 2. * _beta) * pow(x(1), 2.);
        mat(0,1) = _a * _b * _rho * pow(x(0), _beta) * pow(x(1), 2.);
        mat(1,0) = _a * _b * _rho * pow(x(0), _beta) * pow(x(1), 2.);
        mat(1,1) = pow(_b, 2.) * pow(x(1), 2.);
        return mat;
    }

    Eigen::Matrix<T, 2, 2> calcG(const sde::vector_type<T, 2>& x) const
    {
        Eigen::Matrix<T, 2, 2> mat;
        mat(0,0) = (pow(x(0), -2 * _beta) *  pow(x(1), -2)) 
            / (pow(_a, 2) * (1. - pow(_rho, 2)));
        mat(0,1) = -_rho * pow(x(0), -_beta) * pow(x(1), -2)
            / (_a * _b * (1. - pow(_rho, 2)));
        mat(1,0) = -_rho * pow(x(0), -_beta) * pow(x(1), -2)
            / (_a * _b * (1. - pow(_rho, 2)));
        mat(1,1) = pow(x(1), -2)
            / (pow(_b, 2) * (1. - pow(_rho, 2)));
        return mat;
    }
        
    sde::Tensor<T, 2> calcGDiff(const sde::vector_type<T, 2>& x) const override
    {
        sde::Tensor<T, 2> gDiff;
        const T nu = 1. - pow(_rho, 2.);
        gDiff(0,0,0) = -2. * _beta * pow(x(0), -2.*_beta -1) * pow(x(1), -2.)
            / (pow(_a, 2.) * nu);
        gDiff(0,0,1) = -2. * pow(x(0), -2.*_beta) * pow(x(1), -3.)
            / (pow(_a, 2.) * nu);
        gDiff(0,1,0) = _beta * _rho * pow(x(0), -_beta-1.) * pow(x(1), -2.)
            / (_a * _b * nu);
        gDiff(0,1,1) = 2. * _rho * pow(x(0), -_beta) * pow(x(1), -3)
            / (_a * _b * nu);
        gDiff(1,0,0) = gDiff(0,1,0);
        gDiff(1,0,1) = gDiff(0,1,1);
        gDiff(1,1,0) = 0.;
        gDiff(1,1,1) = -2. * pow(x(1), -3)
            / (pow(_b, 2.) * nu);
        return gDiff;
    }
 
    sde::vector_type<T, 2> calcV0(const sde::vector_type<T, 2>& x) const override
    {
        sde::vector_type<T, 2> v;
        v(0) = 0;
        v(1) = 0;
        return v;
    }

private:
    T _a;
    T _b;
    T _beta;
    T _rho;
    constexpr static std::size_t _bmSize = 2;
    constexpr static std::size_t _stateSize = 2;
};
    
} // namespace sde