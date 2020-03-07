#pragma once
#include "sde.h"
namespace sde
{
class SabrMvn {
public:
    using vector_type = sde::vector_type<T, 2>;
    SabrMvn(
        const T& a,
        const T& b,
        const T& beta,
        const T& rho)
    : _a(a), _b(b), _beta(beta), _rho(rho)
    {
        _gamma(0,0,1) = -_b * pow(1.-(_rho * _rho), 0.5);
        _gamma(0,1,0) = _gamma(0,0,1);
        _gamma(1,1,0) = -_b * _rho;
        _gamma(1,0,1) = _gamma(1,1,0);
    }

    std::function<vector_type(const vector_type&)> getV0() const
    {
        auto vec0 = [this](const vector_type& x) {
            vector_type v;
            v(0) = -0.5 * (_a * _a * _beta * pow(x(1),2) * pow(x(0), (2. * _beta -1.)) + _a * _b * _rho * x(1) * pow(x(0), _beta));
            v(1) = -0.5 * _b * _b * x(1);
            return v;
        };
        return vec0;
    }

    std::vector<std::function<lifted_type(const lifted_type&)>>
    getLiftedV(const vector_type& bm) const
    {
        auto&& vec = [this, &bm](const lifted_type& x){
            lifted_type lv0;
            {
                //x(0)=x1, x(1)=x2, x(2)=e11, x(3)=e12,x(4)=e21,x(5)=e22
                const auto v0 = x(2) * _a * pow(x(0), _beta) * x(1);
                auto v1 = x(2) * _b * _rho * x(1) + x(4) * _b * (1. - _rho * _rho)  * x(1);
                const auto n0 = x(2) * x(2) * _gamma(0,0,0) + x(2) * x(4) * _gamma(0,1,0) + x(4) * x(2) * _gamma(1,0,0) + x(4) * x(4) * _gamma(1,1,0);
                auto n1 = x(2) * x(2) * _gamma(0,0,1) + x(2) * x(4) * _gamma(0,1,1) + x(4) * x(2) * _gamma(1,0,1) + x(4) * x(4) * _gamma(1,1,1);
                auto n2 = x(2) * x(3) * _gamma(0,0,0) + x(2) * x(5) * _gamma(0,1,0) + x(4) * x(3) * _gamma(1,0,0) + x(4) * x(5) * _gamma(1,1,0);
                auto n3 = x(2) * x(3) * _gamma(0,0,1) + x(2) * x(5) * _gamma(0,1,1) + x(4) * x(3) * _gamma(1,0,1) + x(4) * x(5) * _gamma(1,1,1);
                lv0 << v0, v1, n0, n1, n2, n3;
            }
           
            lifted_type lv1;
            {
                //x(0)=x1, x(1)=x2, x(2)=e11, x(3)=e12,x(4)=e21,x(5)=e22
                auto&& v0 = x(3) * _a * pow(x(0), _beta) * x(1);
                auto&& v1 = x(3) * _b * _rho * x(1) + x(5) * _b * sqrt(1. - _rho * _rho)  * x(1);
                auto&& n0 = x(3) * x(2) * _gamma(0,0,0) + x(3) * x(4) * _gamma(0,1,0) + x(5) * x(2) * _gamma(1,0,0) + x(5) * x(4) * _gamma(1,1,0);
                auto&& n1 = x(3) * x(2) * _gamma(0,0,1) + x(3) * x(4) * _gamma(0,1,1) + x(5) * x(2) * _gamma(1,0,1) + x(5) * x(4) * _gamma(1,1,1);
                auto&& n2 = x(3) * x(3) * _gamma(0,0,0) + x(3) * x(5) * _gamma(0,1,0) + x(5) * x(3) * _gamma(1,0,0) + x(5) * x(5) * _gamma(1,1,0);
                auto&& n3 = x(3) * x(3) * _gamma(0,0,1) + x(3) * x(5) * _gamma(0,1,1) + x(5) * x(3) * _gamma(1,0,1) + x(5) * x(5) * _gamma(1,1,1);
                lv1 << v0, v1, n0, n1, n2, n3;
            }
            return lv0 * bm(0) + lv1 * bm(1);
        };

        return std::vector<std::function<lifted_type(const lifted_type&)>> 
            {vec};
    }

private:
    T _a;
    T _b;
    T _beta;
    T _rho;
    constexpr static int _bmSize = 2;
    constexpr static int _stateSize = 2;
    sde::Tensor<T, 2> _gamma;
};
}