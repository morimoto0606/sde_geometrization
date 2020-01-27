#pragma once
#include <codi.hpp>
#include "sde.h"
#include "RungeKutta.hpp"
#include "VectorField.hpp"

namespace sde {

template <typename R, typename RD, std::size_t Size>
class StochasticLift {
public:
    StochasticLift(
        double stepSize,
        const RungeKutta<double, R>& rk,
        const RungeKutta<codi::RealReverse, RD>& rkDiff,
        const VectorField<double, Size>& vecField,
        const VectorField<codi::RealReverse, Size>& vecFieldDiff,
        const sde::vector_type<double, Size>& ini):
    _stepSize(stepSize), 
    _rk(rk.clone()),
    _rkDiff(rkDiff.clone()),
    _vecField(vecField.clone()),
    _vecFieldDiff(vecFieldDiff.clone()),
    _ini(ini) {}

    typename VectorField<codi::RealReverse, Size>::lifted_type 
    getLiftedIni(const sde::vector_type<double, Size>& ini) const 
    {
        typename VectorField<codi::RealReverse, Size>::lifted_type liftedIni;
        for (int i = 0; i < Size; ++i) {
            liftedIni(i) = codi::RealReverse(_ini(i));
        }
        const auto id = Eigen::MatrixXd::Identity(Size, Size);
        for (int i = 0; i < Size; ++i) {
            for (int j = 0; j < Size; ++j) {
                liftedIni(Size + Size * i + j) = codi::RealReverse(id(i, j));
            }
        }
        return liftedIni;
    }

    typename VectorField<double, Size>::lifted_type 
    evolveJacobiInv(
        const sde::vector_type<codi::RealReverse, Size>& ini,
        const std::vector<codi::RealReverse>& bm) const  
    {
        /*
        dX(t,x)j=sum_{i=1}^N V_i(X(t)) \cir dB^i(t),X(0)=x
        J(t)=dX/dx
        Calculate J^{-1}(t) by AAD for one sample rand normals [z1,..,zN]
        */
        codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
        tape.setActive();
        tape.registerInput(ini);
        const auto liftedIni = getLiftedIni(_ini);
        const auto flow = _rkDiff->solveIterative(1.0, _vecFieldDiff->getLiftedV(bm), liftedIni);
        tape.registerOutput(flow);
        tape.setPassive();
        auto jac = Eigen::MatrixXd::Zero(Size, Size);
        for (int i = 0; i < Size; ++i) {
            flow(i).setGradient(1.0);
            for (int j = 0; j < Size; ++j) {
                jac(i, j) = ini(j).getGradient();
            }
        }
        return jac.inverse();
    }

private:
    double _stepSize;
    std::unique_ptr<RungeKutta<double, R>> _rk;
    std::unique_ptr<RungeKutta<codi::RealReverse, RD>> _rkDiff;
    std::unique_ptr<VectorField<double, Size>> _vecField;
    std::unique_ptr<VectorField<codi::RealReverse, Size>> _vecFieldDiff;
    sde::vector_type<double, Size> _ini;

};


}//namespace sde 