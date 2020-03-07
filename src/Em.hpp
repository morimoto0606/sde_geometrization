#pragma once
#include <Eigen/Dense>
#include "sde.h"
#include "Scheme.hpp"
#include "RndNormal.hpp"
#include "VectorFieldSabr.hpp"

namespace sde {

class SabrEm : public Scheme<2> {
public:
    SabrEm(
        double a,
        double b,
        double beta,
        double rho): 
        _a(a), _b(b), _beta(beta), _rho(rho),
        _vecField(std::make_unique<Sabr>(a, b, beta, rho))
    {
    }

    sde::vector_type<double, 2> evolve(
        const sde::vector_type<double, 2>& x,
        const sde::vector_type<double, 2>& dB,
        double dt) const override
    {
        auto _drift = _vecField->getV0<double>();


        sde::vector_type<double, 2> y;
        y(0) = x(0) + _a * x(1) * pow(x(0), _beta) * dB(0);
        y(1) = x(1) + _b * x(1) * (_rho * dB(0) + sqrt(1.- _rho * _rho) * dB(1));
        auto drift = _drift(x) * dt;
        auto ret = y - drift;
        //std::cout << "drift "  << drift.transpose() << std::endl;
        return ret;
    }

private:
    double _a;
    double _b;
    double _beta;
    double _rho;
    std::unique_ptr<Sabr> _vecField;
};


class LiftedSabrEm : public Scheme<6> {
public:
    LiftedSabrEm(
        double a,
        double b,
        double beta,
        double rho): 
        _a(a), _b(b), _beta(beta), _rho(rho),
        _vecField(std::make_unique<Sabr>(a, b, beta, rho))
    {
    }

    sde::vector_type<double, 6> evolve(
        const sde::vector_type<double, 6>& x,
        const sde::vector_type<double, 6>& dB,
        double dt) const override
    {
        const auto _drift = _vecField->getV0<double>();
        sde::vector_type<double, 2> x2 = x(Eigen::seqN(0,2));
        const auto gamma = _vecField->calcGamma(x2);
        std::vector<Tensor<double, 2>> dGamma;
        const double epsilon = 0.01;
        for (int alpha = 0; alpha < 2; ++alpha) {
            sde::vector_type<double, 2> xPlus = x(Eigen::seqN(0, 2));
            sde::vector_type<double, 2> xMinus = x(Eigen::seqN(0, 2));
            xPlus(alpha) =  xPlus(alpha) + epsilon;
            xMinus(alpha) = xMinus(alpha) - epsilon;
            dGamma.push_back((_vecField->calcGamma(xPlus) - _vecField->calcGamma(xMinus)) / (2. * epsilon));
        }

        Eigen::Matrix2d e = Eigen::MatrixXd::Zero(2,2);
        e(0, 0) = x(2);
        e(0, 1) = x(3);
        e(1, 0) = x(4);
        e(1, 1) = x(5);
        sde::vector_type<double, 6> y;
        for (int i = 0; i < 2; ++i) {
            for (int alpha = 0; alpha < 2; ++alpha) {
                y(i) = x(i) + e(i, alpha) * dB(alpha);
                for (int k = 0; k < 2; ++k) {
                    for (int l = 0; l < 2; ++l) {
                        y(i) -= 0.5 * gamma(k, l, i) * e(k, alpha) * e(l, alpha) * dt;
                    }
                }
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (int beta = 0; beta < 2; ++beta) {
                for (int k = 0; k < 2; ++k) {
                    for (int l = 0; l < 2; ++l) {
                        for (int alpha = 0; alpha < 2; ++alpha) {
                            y(2 + 2 * i + beta) = x(2 + 2 * i + beta) - gamma(k, l, i) * e(k, alpha) * e(l, beta) * dB(alpha)
                            -0.5 * dGamma[alpha](k, l, i) * e(k, alpha) * e(l, beta) * dt;
                            for (int m = 0; m < 2; ++m) {
                                for (int n = 0; n < 2; ++n) {
                                   y(2 + 2 * i + beta) += 0.5 * (gamma(k,l,i) * gamma(m,n,k) * e(l,beta) * e(m,alpha) * e(n,alpha)
                                                               + gamma(k,l,i) * gamma(m,n,l) * e(k,alpha) * e(m,alpha) * e(n,beta)) * dt;
                                }
                            }
                        }
                    }
                }
            }
        }

        //std::cout << "drift "  << drift.transpose() << std::endl;
        return y;
 
    }

private:
    double _a;
    double _b;
    double _beta;
    double _rho;
    std::unique_ptr<Sabr> _vecField;
};



}