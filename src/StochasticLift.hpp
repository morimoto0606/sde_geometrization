#pragma once
#include <codi.hpp>
#include <Eigen/Dense>
#include "sde.h"
#include "RungeKutta.hpp"
#include "VectorField.hpp"
#include "RndNormal.hpp"

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

        auto zeta = _rk->solve(_stepSize, vecFieldZeta, prev);
        return zeta;
    }

    sde::vector_type<double, Size> evolveX(
        const sde::vector_type<double, Size>& prev,
        const sde::vector_type<double, Size>& bm) const 
    {
        const sde::vector_type<double, Size> zeta = this->evolveZeta(prev, bm);
        const sde::lifted_type<double, Size> liftedIni = this->getLiftedIni<double>(zeta);
        const sde::lifted_type<double, Size> liftedX
            = _rk->solveIterative(1.0, _vecField->getLiftedV(bm), liftedIni);
        const sde::vector_type<double, Size> x = liftedX(Eigen::seqN(0, Size));
        return x;
    }


    sde::vector_type<double, Size> generateOnePath(
        std::size_t numSteps,
        const Eigen::MatrixXd& bm) const
    {
        sde::vector_type<double, Size> x = _ini;
        for (std::size_t i = 0; i < numSteps; ++i) {
            x = this->evolveX(x, bm.col(i));
        }
        return x;
    }

    Eigen::MatrixXd generatePaths(
        const std::size_t numSteps,
        const std::size_t pathNum,
        const RndNormal& generator) const 
    {   
        Eigen::MatrixXd path(Size, pathNum);
        for (int p = 0; p < pathNum; ++p) {
            const Eigen::MatrixXd& normal = generator.get(Size, numSteps);
            const Eigen::MatrixXd& bm = sqrt(_stepSize) * normal;
            const sde::vector_type<double, Size>& x = this->generateOnePath(numSteps, bm);
            path.col(p) = x;
        }
        return path;
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