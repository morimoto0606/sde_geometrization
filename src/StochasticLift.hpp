#pragma once
#include <codi.hpp>
#include <Eigen/Dense>
#include "sde.h"
#include "Scheme.hpp"
#include "RungeKutta.hpp"
#include "VectorField.hpp"
#include "RndNormal.hpp"

namespace sde {

template <typename R, typename V, std::size_t Size>
class StochasticLift : public Scheme<Size> {
public:
    StochasticLift(
        const RungeKutta<R>& rk,
        const VectorField<V, Size>& vecField,
        int numStepRk):
    _rk(rk.clone()),
    _vecField(vecField.clone()),
    _numStepRk(numStepRk){}

    template <typename T, typename U>
    sde::lifted_type<T, Size> getLiftedIni(
        const sde::vector_type<U, Size>& ini) const 
    {
        sde::lifted_type<T, Size> liftedIni;
        for (int i = 0; i < Size; ++i) {
            liftedIni(i) = ini(i);
        }
        const auto id = Eigen::MatrixXd::Identity(Size, Size);
        auto vecField = _vecField->calcV(ini);
        for (int i = 0; i < Size; ++i) {
            for (int j = 0; j < Size; ++j) {
                liftedIni(Size + Size * i + j) = vecField(i, j);
            }
        }
        return liftedIni;
    }

    template <typename T>
    sde::vector_type<T, Size> evolveXi(
        const sde::vector_type<T, Size>& prev,
        const sde::vector_type<double, Size>& bm) const  
    {
        /*
        dXi(t,x)j=sum_{i=1}^N V_i(Xi(t))dB^i(t)-1/2g^{kl}Gamma_{kl}^idt ,Xi(0)=x
        */
        auto liftedIni = this->getLiftedIni<T>(prev);
        auto liftedX
            = _rk->solveIterative(1.0, _numStepRk, _vecField->template getLiftedV<T>(bm), liftedIni);
        auto x = liftedX(Eigen::seqN(0, Size));
        return x;
    }
 
    Eigen::Matrix<double, Size, Size> evolveJacobiInv(
        const sde::vector_type<double, Size>& prev,
        const sde::vector_type<double, Size>& bm) const  
    {
        /*
        dX(t,x)j=sum_{i=1}^N V_i(X(t)) \cir dB^i(t),X(0)=x
        J(t)=dX/dx
        Calculate J^{-1}(t) by AAD for one sample rand normals [z1,..,zN]
        */
        auto liftedIni = getLiftedIni<codi::RealReverse>(prev);
        codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
        tape.setActive();
        for (int i = 0; i < Size; ++i) {
            tape.registerInput(liftedIni(i));
        }
        auto flow = _rk->solveIterative(1.0, 1, _vecField->template getLiftedV<codi::RealReverse>(bm), liftedIni);
        for (int i = 0; i < Size; ++i) {
            tape.registerOutput(flow(i));
        }
        tape.setPassive();

        Eigen::Matrix<double, Size, Size> jac;
        for (int i = 0; i < Size; ++i) {
            flow(i).setGradient(1.0);
            for (int j = 0; j < Size; ++j) {
                tape.evaluate();
                jac(i, j) = liftedIni(j).getGradient();
            }
            tape.clearAdjoints();
        }
 
        tape.reset();
        return jac.inverse();
    }

    sde::vector_type<double, Size> evolveZeta(
        const sde::vector_type<double, Size>& prev,
        const sde::vector_type<double, Size>& bm) const 
    {
        const Eigen::Matrix<double, Size, Size> jacobi 
            = this->evolveJacobiInv(prev, bm);
        const sde::function_type<sde::vector_type<double, Size>> v0
            = _vecField->template getV0<double>();

        auto vecFieldZeta = [this, &bm, &jacobi, &v0](const sde::vector_type<double, Size>& x)
        {
            const sde::lifted_type<double, Size> liftedIni
                = this->getLiftedIni<double>(x);
            const sde::lifted_type<double, Size> liftedFlow = _rk->solveIterative(
                1.0,
                _numStepRk, 
                _vecField->template getLiftedV<double>(bm),
                liftedIni);
            const sde::vector_type<double, Size> flow = liftedFlow(Eigen::seqN(0, Size));
            const sde::vector_type<double, Size> m = jacobi * v0(flow);
 
            return m;
        };
        
        std::vector<sde::func_ptr_type<sde::vector_type<double, Size>>> 
        vecFieldZetaPtr{std::make_shared<sde::function_type<sde::vector_type<double, Size>>>(vecFieldZeta)};

        auto zeta = _rk->solveIterative(1.0, _numStepRk, vecFieldZetaPtr, prev);
        return zeta;
    }


    sde::vector_type<double, Size> evolve(
        const sde::vector_type<double, Size>& prev,
        const sde::vector_type<double, Size>& bm) const 
    {
        const sde::vector_type<double, Size> zeta = this->evolveZeta(prev, bm);
        return evolveXi(zeta, bm);
    }

private:
    std::unique_ptr<RungeKutta<R>> _rk;
    std::unique_ptr<VectorField<V, Size>> _vecField;
    int _numStepRk;
};

}//namespace sde 