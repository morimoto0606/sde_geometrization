#pragma once
#include "sde.h"
namespace sde
{

template <typename T, std::size_t Size>
class VectorField {
public:
    using vector_type = sde::vector_type<T, Size>;
    using lifted_type = sde::lifted_type<T, Size>;

    virtual ~VectorField() = default;
    virtual std::unique_ptr<VectorField<T, Size>> clone() const = 0;
    virtual std::function<vector_type(const vector_type&)> getV0() const = 0;
    virtual std::vector<std::shared_ptr<const std::function<lifted_type(const lifted_type&)>>>
    getLiftedV(const sde::vector_type<double, Size>& bm) const = 0;
 
};

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
        _gamma(0,0,1) = -_b * pow(1.-(_rho * _rho), 0.5);
        _gamma(0,1,0) = _gamma(0,0,1);
        _gamma(1,1,0) = -_b * _rho;
        _gamma(1,0,1) = _gamma(1,1,0);
    }

    std::unique_ptr<VectorField<T, 2>> clone() const override 
    {
        return std::make_unique<Sabr<T>>(*this);
    }

    virtual sde::function_type<vector_type> getV0() const override
    {
        auto vec0 = [this](const vector_type& x) {
            vector_type v;
            v(0) = -0.5 * (_a * _a * _beta * pow(x(1),2) * pow(x(0), (2. * _beta -1.)) + _a * _b * _rho * x(1) * pow(x(0), _beta));
            v(1) = -0.5 * _b * _b * x(1);
            return v;
        };
        return vec0;
    }

    virtual std::vector<sde::func_ptr_type<lifted_type>>
    getLiftedV(const sde::vector_type<double, 2>& bm) const override
    {
        auto vec1 = [this, bm](const lifted_type& x){
            //x(0)=x1, x(1)=x2, x(2)=e11, x(3)=e12,x(4)=e21,x(5)=e22
            const auto v0 = x(2) * _a * pow(x(0), _beta) * x(1);
            const auto v1 = x(2) * _b * _rho * x(1) + x(4) * _b * (1. - _rho * _rho)  * x(1);
            const auto n0 = x(2) * x(2) * _gamma(0,0,0) + x(2) * x(4) * _gamma(0,1,0) + x(4) * x(2) * _gamma(1,0,0) + x(4) * x(4) * _gamma(1,1,0);
            const auto n1 = x(2) * x(2) * _gamma(0,0,1) + x(2) * x(4) * _gamma(0,1,1) + x(4) * x(2) * _gamma(1,0,1) + x(4) * x(4) * _gamma(1,1,1);
            const auto n2 = x(2) * x(3) * _gamma(0,0,0) + x(2) * x(5) * _gamma(0,1,0) + x(4) * x(3) * _gamma(1,0,0) + x(4) * x(5) * _gamma(1,1,0);
            const auto n3 = x(2) * x(3) * _gamma(0,0,1) + x(2) * x(5) * _gamma(0,1,1) + x(4) * x(3) * _gamma(1,0,1) + x(4) * x(5) * _gamma(1,1,1);
            lifted_type v;
            v << v0, v1, n0, n1, n2, n3;
            lifted_type w = bm(0) * v;
            return w;
        };

        auto vec2 = [this, bm](const lifted_type& x){
            //x(0)=x1, x(1)=x2, x(2)=e11, x(3)=e12,x(4)=e21,x(5)=e22
            const auto v0 = x(3) * _a * pow(x(0), _beta) * x(1);
            const auto v1 = x(3) * _b * _rho * x(1) + x(5) * _b * sqrt(1. - _rho * _rho)  * x(1);
            const auto n0 = x(3) * x(2) * _gamma(0,0,0) + x(3) * x(4) * _gamma(0,1,0) + x(5) * x(2) * _gamma(1,0,0) + x(5) * x(4) * _gamma(1,1,0);
            const auto n1 = x(3) * x(2) * _gamma(0,0,1) + x(3) * x(4) * _gamma(0,1,1) + x(5) * x(2) * _gamma(1,0,1) + x(5) * x(4) * _gamma(1,1,1);
            const auto n2 = x(3) * x(3) * _gamma(0,0,0) + x(3) * x(5) * _gamma(0,1,0) + x(5) * x(3) * _gamma(1,0,0) + x(5) * x(5) * _gamma(1,1,0);
            const auto n3 = x(3) * x(3) * _gamma(0,0,1) + x(3) * x(5) * _gamma(0,1,1) + x(5) * x(3) * _gamma(1,0,1) + x(5) * x(5) * _gamma(1,1,1);
            lifted_type v;
            v << v0, v1, n0, n1, n2, n3;
            lifted_type w = bm(1) * v;
            return w;
        };

        return std::vector<sde::func_ptr_type<lifted_type>>{
            std::make_shared<sde::function_type<lifted_type>>(vec1),
            std::make_shared<sde::function_type<lifted_type>>(vec2)
        };
    }

private:
    T _a;
    T _b;
    T _beta;
    T _rho;
    constexpr static std::size_t _bmSize = 2;
    constexpr static std::size_t _stateSize = 2;
    sde::Tensor<T, 2> _gamma;
};
    
} // namespace sde