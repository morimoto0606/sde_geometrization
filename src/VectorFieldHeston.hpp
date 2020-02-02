#pragma once
#include "sde.h"
#include "VectorField.hpp"
namespace sde
{

template <typename T>
class Heston : public VectorField<T, 2> {
public:
    using vector_type = sde::vector_type<T, 2>;
    using lifted_type = sde::lifted_type<T, 2>;
 
   Heston(
        const T& xi,
        const T& rho)
    : _xi(xi), _rho(rho)
    {

    }

    std::unique_ptr<VectorField<T, 2>> clone() const override 
    {
        return std::make_unique<Heston<T>>(*this);
    }


    Eigen::Matrix<T, 2, 2> calcGInv(const sde::vector_type<T, 2>& x) const override
    {
        const T nu = 1. - pow(_rho, 2.);
        Eigen::Matrix<T, 2, 2> mat;
        mat(0,0) = _rho / nu * pow(x(0), -2) + pow(x(0), -2) / x(1);
        mat(0,1) = -_rho / (_xi * nu) * pow(x(1), -0.5) / x(1);
        mat(1,0) = m(0,1);
        mat(1,1) = 1.0 / (pow(_xi) * nu * x(1));
        return mat;
    }
       
    sde::Tensor<T, 2> calcGDiff(const sde::vector_type<T, 2>& x) const override
    {
        const T nu = 1. - pow(_rho, 2.);
        sde::Tensor<T, 2> gDiff;
        const T nu = 1. - pow(_rho, 2.);
        gDiff(0,0,0) = -2. * _rho / nu * pow(x(0), -3.) -2. * pow(x(0), -3.) * pow(x(1), -1.);
        gDiff(0,0,1) = -pow(x(0) * x(1), -2.);
        gDiff(0,1,0) = _rho / (_xi * nu) * pow(x(0), -2.) * pow(x(1), -0.5);
        gDiff(0,1,1) = 0.5 * _rho /  (_xi * nu) * pow(x(0), -1.) * pow(x(1), -1.5);
        gDiff(1,0,0) = gDiff(0,1,0):
        gDiff(1,0,1) = gDiff(0,1,1);
        gDiff(1,1,0) = 0.;
        gDiff(1,1,1) = -pow(x(1), -2.) / (pow(xi, 2.) * nu);
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
    T _xi;
    T _rho;
    constexpr static std::size_t _bmSize = 2;
    constexpr static std::size_t _stateSize = 2;
};
    
} // namespace sde