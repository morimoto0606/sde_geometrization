#pragma once
#include <random>
#include <Eigen/Core>

namespace sde {

class RndNormal {
public:
    virtual ~RndNormal() = default;

    virtual Eigen::MatrixXd get(
        const std::size_t dim,
        const std::size_t size) const = 0;

};

class MtNormal : public RndNormal {
public:
    MtNormal(const std::size_t seed): 
        _mt(std::make_unique<std::mt19937>(seed)), 
        _norm(std::make_unique<std::normal_distribution<>>(0.0, 1.0))
    {
    }

    Eigen::MatrixXd get(
        const std::size_t dim,
        const std::size_t size) const override
    {
        Eigen::MatrixXd ret(dim, size);
        for (std::size_t j = 0; j < size; ++j) {
            for (std::size_t i = 0; i < dim; ++i) {
                ret(i, j) = (*_norm)(*_mt);
            }
        }
        return ret;
    }

private:
    std::unique_ptr<std::mt19937> _mt;
    std::unique_ptr<std::normal_distribution<>> _norm;
 
};

}