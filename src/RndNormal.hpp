#pragma once
#include <random>
#include <Eigen/Core>
#include "sobol.h"

namespace sde {

class RndNormal {
public:
    virtual ~RndNormal() = default;

    virtual std::vector<Eigen::MatrixXd> get(
        const std::size_t pathNum,
        const std::size_t rowSize,
        const std::size_t colSize) const = 0;

};

class MtNormal : public RndNormal {
public:
    MtNormal(const std::size_t seed): 
        _mt(std::make_unique<std::mt19937>(seed)), 
        _norm(std::make_unique<std::normal_distribution<>>(0.0, 1.0))
    {
    }

    std::vector<Eigen::MatrixXd> get(
        const std::size_t pathNum,
        const std::size_t rowSize,
        const std::size_t colSize) const override
    {
        std::vector<Eigen::MatrixXd> ret;
        ret.reserve(pathNum);
        for (std::size_t p = 0; p < pathNum; ++p) {
            Eigen::MatrixXd x(rowSize, colSize);
            for (std::size_t i = 0; i < rowSize; ++i) {
                for (std::size_t j = 0; j < colSize; ++j) {
                    x(i, j) = (*_norm)(*_mt);
                }
            }
            ret.emplace_back(x);
        }
        return ret;
    }

private:
    std::unique_ptr<std::mt19937> _mt;
    std::unique_ptr<std::normal_distribution<>> _norm;
};

class SobolNormal : public RndNormal {
public:
    SobolNormal(int initialSkip): _initialSkip(initialSkip){}
    std::vector<Eigen::MatrixXd> get(
        const std::size_t pathNum,
        const std::size_t rowSize,
        const std::size_t colSize) const override
    {
        std::vector<Eigen::MatrixXd> ret;
        ret.reserve(pathNum);
        auto normals = sobol_points_normal(pathNum, rowSize * colSize, _initialSkip);
        for (std::size_t p = 0; p < pathNum; ++p) {
            Eigen::MatrixXd x(rowSize, colSize);
            auto normal = normals.row(p);
            for (std::size_t i = 0; i < rowSize; ++i) {
                for (std::size_t j = 0; j < colSize; ++j) {
                    x(i, j) = normal(i * colSize + j);
                }
            }
            ret.emplace_back(x);
        }
        return ret;
    }
  
private:
    int _initialSkip;
};

}