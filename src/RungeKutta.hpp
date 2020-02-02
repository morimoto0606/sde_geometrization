#pragma once
#include <iostream>
#include <memory>
#include <Eigen/Core>
#include "sde.h"

namespace sde
{
template <typename T, typename Derived>
class RungeKutta {
public:

    virtual ~RungeKutta() = default;
    virtual std::unique_ptr<RungeKutta<T, Derived>> clone() const = 0;

    template <typename V, typename F>
    V solve(
        const T& h,
        const F& vecfield,
        const V& ini_val) const {
            return dynamic_cast<const Derived*>(this)->solve(h, vecfield, ini_val);
        }
    
    template <typename V, typename F>
    V solveIterative(
        const T& h,
        const std::vector<std::shared_ptr<const F>>& vecfields,
        const V& ini_val) const {
            return dynamic_cast<const Derived*>(this)->solveIterative(h, vecfields, ini_val);
        }
};

template <typename T>
class RungeKutta5 : public RungeKutta<T, RungeKutta5<T>> {
public:

    RungeKutta5() {
        _a(1,0) = 2./5;
        _a(2,0) = 11./64;
        _a(2,1) = 5./64;
        _a(3,2) = 0.5;
        _a(4,0) = 3./64;
        _a(4,1) = -15./64;
        _a(4,2) = 3./8;
        _a(4,3) = 9./16;
        _a(5,1) = 5./7;
        _a(5,2) = 6./7;
        _a(5,3) = -12./7;
        _a(5,4) = 8./7;

        _b << 7.,0.,32.,12.,32.,7.;
        _b /= 90.;

    }

    std::unique_ptr<RungeKutta<T, RungeKutta5<T>>> clone() const override
    {
        return std::make_unique<RungeKutta5<T>>(*this);
    }

private:
        Eigen::Matrix<T, 6, 6> _a;
        Eigen::Matrix<T, 6, 1> _b;

public:

    template <typename V, typename F>
    V solve(
        const T& h,
        const F& vecfield,
        const V& ini_val) const
    {   
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> k(ini_val.size(), 6);

        k.col(0) = vecfield(ini_val);
        std::cout << " k.col0 = " << k.col(0) << std::endl;
        k.col(1) = vecfield(ini_val + h * (_a(1,0) * k.col(0)));
        std::cout << " k.col1 = " << k.col(1) << std::endl;

        k.col(2) = vecfield(ini_val + h * (_a(2,0) * k.col(0) + _a(2,1) * k.col(1)));
        std::cout << " k.col2 = " << k.col(2) << std::endl;

        k.col(3) = vecfield(ini_val + h * (_a(3,0) * k.col(0) + _a(3,1) * k.col(1) + _a(3,2) * k.col(2)));
        std::cout << " k.col3 = " << k.col(3) << std::endl;

        k.col(4) = vecfield(ini_val + h * (_a(4,0) * k.col(0) + _a(4,1) * k.col(1) + _a(4,2) * k.col(2) + _a(4,3) * k.col(3)));
        std::cout << " k.col4 = " << k.col(4) << std::endl;

        k.col(5) = vecfield(ini_val + h * (_a(5,0) * k.col(0) + _a(5,1) * k.col(1) + _a(5,2) * k.col(2) + _a(5,3) * k.col(3) + _a(5,4) * k.col(4)));
        std::cout << " k.col5 = " << k.col(5) << std::endl;

        auto ret = ini_val + h * k * _b;
        std::cout << " ret = " << vecfield(ret) << std::endl;


        return ret;
    }
    
    template <typename V, typename F>
    V solveIterative(
        const T& h,
        const std::vector<std::shared_ptr<const F>>& vecfields,
        const V& ini_val) const
    {
        auto x = ini_val;
        for (auto v: vecfields) {
            x = this->solve(h, *v, x);
        }
        return x;
    }
};
}