#pragma once
#include "sde.h"
#include "VectorField.hpp"
namespace sde
{
class Heston : public VectorField<Heston, 2> {
public:
 
   Heston(
        double mu,
        double kappa,
        double theta,
        double xi,
        double rho)
    : _mu(mu), _kappa(kappa), _theta(theta), _xi(xi), _rho(rho)
    {
    }

    std::unique_ptr<VectorField<Heston, 2>> clone() const override 
    {
        return std::make_unique<Heston>(*this);
    }

    template <typename T>
    Eigen::Matrix<T, 2, 2> calcV(const sde::vector_type<T, 2>& x) const
    {
        Eigen::Matrix<T, 2, 2> mat;
        mat(0,0) = sqrt(x(1)) * x(0);
        mat(0,1) = 0.;
        mat(1,0) = _xi * _rho * sqrt(x(1));
        mat(1,1) = _xi * sqrt(1. - pow(_rho, 2.)) * sqrt(x(1));
        return mat;
    }

    template <typename T>
    sde::Tensor<T, 2> calcGDiff(const sde::vector_type<T, 2>& x) const
    {
        const T nu = 1. - pow(_rho, 2.);
        sde::Tensor<T, 2> gDiff;
        gDiff(0,0,0) = -2. * _rho / nu * pow(x(0), -3.) * pow(x(1), -1.); 
        gDiff(0,0,1) = - _rho / nu * pow(x(0) * x(1), -2.);
        gDiff(0,1,0) = _rho / (_xi * nu) * pow(x(0), -2.) * pow(x(1), -1.);
        gDiff(0,1,1) = _rho /  (_xi * nu) * pow(x(0), -1.) * pow(x(1), -2.);
        gDiff(1,0,0) = gDiff(0,1,0);
        gDiff(1,0,1) = gDiff(0,1,1);
        gDiff(1,1,0) = 0.;
        gDiff(1,1,1) = -pow(x(1), -2.) / (pow(_xi, 2.) * nu);
        return gDiff;
    }
    
    template <typename T>
    sde::vector_type<T, 2> calcV0(const sde::vector_type<T, 2>& x) const
    {
        sde::vector_type<T, 2> v;
        v(0) = 0;
        v(1) = 0;
        return v;
    }

private:
    double _mu;
    double _kappa;
    double _theta;
    double _xi;
    double _rho;
    constexpr static std::size_t _bmSize = 2;
    constexpr static std::size_t _stateSize = 2;
};
  
  
} // namespace sde