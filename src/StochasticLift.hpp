#pragma once
#include <codi.hpp>
#include <Eigen/Dense>
#include "sde.h"
#include "Scheme.hpp"
#include "RungeKutta.hpp"
#include "VectorField.hpp"
#include "RndNormal.hpp"

namespace sde {

template <typename R, typename RD, std::size_t Size>
class StochasticLift : public Scheme<Size> {
public:
    StochasticLift(
        const RungeKutta<double, R>& rk,
        const RungeKutta<codi::RealReverse, RD>& rkDiff,
        const VectorField<double, Size>& vecField,
        const VectorField<codi::RealReverse, Size>& vecFieldDiff):
    _rk(rk.clone()),
    _rkDiff(rkDiff.clone()),
    _vecField(vecField.clone()),
    _vecFieldDiff(vecFieldDiff.clone()){}

    template <typename T>
    sde::lifted_type<T, Size> getLiftedIni(
        const sde::vector_type<double, Size>& ini) const 
    {
        sde::lifted_type<T, Size> liftedIni;
        for (int i = 0; i < Size; ++i) {
            liftedIni(i) = ini(i);
        }
        const auto id = Eigen::MatrixXd::Identity(Size, Size);
        for (int i = 0; i < Size; ++i) {
            for (int j = 0; j < Size; ++j) {
                liftedIni(Size + Size * i + j) = id(i, j);
            }
        }
        return liftedIni;
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
        std::cout << "bm = "  << bm(0) << ", " << bm(1) << std::endl;
        auto flow = _rkDiff->solveIterative(1.0, _vecFieldDiff->getLiftedV(bm), liftedIni);
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
            = _vecField->getV0();

        auto vecFieldZeta = [this, &bm, &jacobi, &v0](const sde::vector_type<double, Size>& x)
        {
            const sde::lifted_type<double, Size> liftedIni
                = this->getLiftedIni<double>(x);
            const sde::lifted_type<double, Size> liftedFlow = _rk->solveIterative(
                1.0, 
                _vecField->getLiftedV(bm),
                liftedIni);
            //std::cout << "liftedFlow = " << liftedFlow << std::endl;
            const sde::vector_type<double, Size> flow = liftedFlow(Eigen::seqN(0, Size));
            //std::cout << "flow = " << flow << std::endl;
            const sde::vector_type<double, Size> m = jacobi * v0(flow);
            //std::cout << "jacobi = " << jacobi << std::endl;
            //std::cout << "v0(flow)= " << v0(flow)<< std::endl;
 
            //std::cout << "m = " << m << std::endl;
            return m;
        };

        auto zeta = _rk->solve(1.0, vecFieldZeta, prev);
        return zeta;
    }

    sde::vector_type<double, Size> evolve(
        const sde::vector_type<double, Size>& prev,
        const sde::vector_type<double, Size>& bm) const override
    {
        const sde::vector_type<double, Size> zeta = this->evolveZeta(prev, bm);
        const sde::lifted_type<double, Size> liftedIni = this->getLiftedIni<double>(zeta);
        const sde::lifted_type<double, Size> liftedX
            = _rk->solveIterative(1.0, _vecField->getLiftedV(bm), liftedIni);
        const sde::vector_type<double, Size> x = liftedX(Eigen::seqN(0, Size));
        return x;
    }


private:
    std::unique_ptr<RungeKutta<double, R>> _rk;
    std::unique_ptr<RungeKutta<codi::RealReverse, RD>> _rkDiff;
    std::unique_ptr<VectorField<double, Size>> _vecField;
    std::unique_ptr<VectorField<codi::RealReverse, Size>> _vecFieldDiff;
};

}//namespace sde 