#pragma once
#include <iostream>
#include <memory>
#include <Eigen/Core>
#include "sde.h"

namespace sde
{
template <typename Derived>
class RungeKutta {
public:

    virtual ~RungeKutta() = default;
    virtual std::unique_ptr<RungeKutta<Derived>> clone() const = 0;

    template <typename V, typename F>
    V solve(
        double h,
        const F& vecfield,
        const V& ini_val) const 
    {
        return dynamic_cast<const Derived*>(this)->solve(h, vecfield, ini_val);
    }
    
    template <typename V, typename F>
    V solveIterative(
        double maturity,
        int numDiscretization,
        const std::vector<std::shared_ptr<const F>>& vecfields,
        const V& ini_val) const
    {
        const double h = maturity / numDiscretization;
        auto x = ini_val;
        for (auto v: vecfields) {
            for (int i = 0; i < numDiscretization; ++i) {
                x = this->solve(h, *v, x);
            }
        }
        return x;
    }

};

class RungeKutta5 : public RungeKutta<RungeKutta5> {
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

    std::unique_ptr<RungeKutta<RungeKutta5>> clone() const override
    {
        return std::make_unique<RungeKutta5>(*this);
    }

private:
        Eigen::Matrix<double, 6, 6> _a;
        Eigen::Matrix<double, 6, 1> _b;

public:

    template <typename V, typename F>
    V solve(
        double h,
        const F& vecfield,
        const V& ini_val) const
    {   
        Eigen::Matrix<typename V::value_type, Eigen::Dynamic, Eigen::Dynamic> 
            k(ini_val.size(), 6);

        k.col(0) = vecfield(ini_val);
        k.col(1) = vecfield(ini_val + h * (_a(1,0) * k.col(0)));
        k.col(2) = vecfield(ini_val + h * (_a(2,0) * k.col(0) + _a(2,1) * k.col(1)));
        k.col(3) = vecfield(ini_val + h * (_a(3,0) * k.col(0) + _a(3,1) * k.col(1) + _a(3,2) * k.col(2)));
        k.col(4) = vecfield(ini_val + h * (_a(4,0) * k.col(0) + _a(4,1) * k.col(1) + _a(4,2) * k.col(2) + _a(4,3) * k.col(3)));
        k.col(5) = vecfield(ini_val + h * (_a(5,0) * k.col(0) + _a(5,1) * k.col(1) + _a(5,2) * k.col(2) + _a(5,3) * k.col(3) + _a(5,4) * k.col(4)));
        auto v = h * (k.col(0) * _b(0) + k.col(1) * _b(1) + k.col(2) * _b(2) + k.col(3) * _b(3) + k.col(4) * _b(4) +  k.col(5) * _b(5));
        auto ret = ini_val + v;
        return ret;
    }
   
};

class RungeKutta4 : public RungeKutta<RungeKutta4> {
public:

    RungeKutta4() {
        _a(1,0) = .5;
        _a(2,1) = .5;
        _a(3,2) = 1.;

        _b << 1.,2.,2.,1.;
        _b /= 6.;

    }

    std::unique_ptr<RungeKutta<RungeKutta4>> clone() const override
    {
        return std::make_unique<RungeKutta4>(*this);
    }

private:
        Eigen::Matrix<double, 4, 4> _a;
        Eigen::Matrix<double, 4, 1> _b;

public:

    template <typename V, typename F>
    V solve(
        double h,
        const F& vecfield,
        const V& ini_val) const
    {   
        Eigen::Matrix<typename V::value_type, Eigen::Dynamic, Eigen::Dynamic> 
            k(ini_val.size(), 4);

        k.col(0) = vecfield(ini_val);
        k.col(1) = vecfield(ini_val + h * (_a(1,0) * k.col(0)));
        k.col(2) = vecfield(ini_val + h * (_a(2,0) * k.col(0) + _a(2,1) * k.col(1)));
        k.col(3) = vecfield(ini_val + h * (_a(3,0) * k.col(0) + _a(3,1) * k.col(1) + _a(3,2) * k.col(2)));
        auto v = h * (k.col(0) * _b(0) + k.col(1) * _b(1) + k.col(2) * _b(2) + k.col(3) * _b(3));
        auto ret = ini_val + v;
        return ret;
    }
    
};

class RungeKutta2 : public RungeKutta<RungeKutta2> {
public:

    RungeKutta2() {
        _a(1,0) = 1.0;
        _b << 1.,1.;
        _b /= 2.;
    }

    std::unique_ptr<RungeKutta<RungeKutta2>> clone() const override
    {
        return std::make_unique<RungeKutta2>(*this);
    }

private:
        Eigen::Matrix<double, 2, 2> _a;
        Eigen::Matrix<double, 2, 1> _b;

public:

    template <typename V, typename F>
    V solve(
        double h,
        const F& vecfield,
        const V& ini_val) const
    {   
        Eigen::Matrix<typename V::value_type, Eigen::Dynamic, Eigen::Dynamic> 
            k(ini_val.size(), 2);

        k.col(0) = vecfield(ini_val);
        k.col(1) = vecfield(ini_val + h * (_a(1,0) * k.col(0)));
        auto v = h * (k.col(0) * _b(0) + k.col(1) * _b(1));
        auto ret = ini_val + v;
        return ret;
    }
    
};
}