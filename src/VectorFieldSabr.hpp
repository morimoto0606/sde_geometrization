#pragma once
#include "sde.h"
#include "VectorField.hpp"
namespace sde
{

class Sabr : public VectorField<Sabr, 2> { 
    public: 
    Sabr( double a, double b,
        double beta,
        double rho)
    : _a(a), _b(b), _beta(beta), _rho(rho)
    {
    }

    std::unique_ptr<VectorField<Sabr, 2>> clone() const override 
    {
        return std::make_unique<Sabr>(*this);
    }

    template <typename T>
    Eigen::Matrix<T, 2, 2> calcV(const sde::vector_type<T, 2>& x) const
    {
        Eigen::Matrix<T, 2, 2> mat;
        mat(0,0) = _a * pow(x(0), _beta) * x(1);
        mat(0,1) = 0.;
        mat(1,0) = _b * _rho * x(1);
        mat(1,1) = _b * sqrt(1. - pow(_rho, 2.)) * x(1);
        return mat;
    }

    template <typename T>
    sde::Tensor<T, 2> calcGDiff(const sde::vector_type<T, 2>& x) const
    {
        sde::Tensor<T, 2> gDiff;
        //const double epsilon = 0.0001;
        //sde::vector_type<T, 2> xPlus = x;
        //sde::vector_type<T, 2> xMinus = x;

        //for (int k = 0; k < 2; ++k) {
        //    xPlus(k) += epsilon;
        //    xMinus(k) -= epsilon;
        //    auto diff =(calcGInv(xPlus) - calcGInv(xMinus)) / (2. * epsilon);
        //    for (int i = 0; i < 2; ++i) {
        //        for (int j = 0; j < 2; ++j) {
        //            gDiff(i, j, k) = diff(i, j);
        //        }
        //    }
        //}
        const T nu = 1. - pow(_rho, 2.);
        gDiff(0,0,0) = -2. * _beta / (pow(_a, 2.) * nu) * pow(x(0), -2.*_beta -1) * pow(x(1), -2.);
        gDiff(0,0,1) = -2. / (pow(_a, 2.) * nu) * pow(x(0), -2.*_beta) * pow(x(1), -3.);
        gDiff(0,1,0) = _beta * _rho / (_a * _b * nu) * pow(x(0), -_beta-1.) * pow(x(1), -2.);
        gDiff(0,1,1) = 2. * _rho / (_a * _b * nu) * pow(x(0), -_beta) * pow(x(1), -3);
        gDiff(1,0,0) = gDiff(0,1,0);
        gDiff(1,0,1) = gDiff(0,1,1);
        gDiff(1,1,0) = 0.;
        gDiff(1,1,1) = -2. / (pow(_b, 2.) * nu) * pow(x(1), -3);
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
    double _a;
    double _b;
    double _beta;
    double _rho;
    constexpr static std::size_t _bmSize = 2;
    constexpr static std::size_t _stateSize = 2;
};
 
} // namespace sde