#pragma once
#include "sde.h"
namespace sde
{

template <typename T>
class Sabr {
public:
    using vector_type = sde::vector_type<T, 2>;
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
  

    std::function<vector_type(const vector_type&)> getV0() const
    {
        auto vec0 = [this](const vector_type& x) {
            vector_type v;
            v(0) = -0.5 * (_a * _a * _beta * pow(x(1),2) * pow(x(0), (2. * _beta -1.)) + _a * _b * _rho * x(1) * pow(x(0), _beta));
            v(1) = -0.5 * _b * _b * x(1);
            return v;
        };
        return vec0;
    }

    

private:
    T _a;
    T _b;
    T _beta;
    T _rho;
    constexpr static int _bmSize = 2;
    constexpr static int _stateSize = 2;
    sde::Tensor<T, 2> _gamma;
};
    
} // namespace sde


//
//class Sabr:
//    """
//    gamma(i,j,k) = ChristrfelSymbol Gamma_{i,j}^k
//    """
//    def __init__(self,
//        a: float,
//        b: float,
//        beta: float,
//        rho: float):
//        _state_size = 2
//        _bm_size = 2
//        _a = a
//        _b = b
//        _beta = beta
//        _rho = rho
//        _gamma = np.zeros((2,2,2))
//        _gamma(0,0,1) = -b * (1.-(rho ** 2.)) ** 0.5
//        _gamma(0,1,0) = _gamma(0,0,1)
//        _gamma(1,1,0) = -b * rho 
//        _gamma(1,0,1) = _gamma(1,1,0)
//        
//
//    def get_state_size(self):
//        return _state_size
//
//    def get_bm_size(self):
//        return _bm_size
//
//    def lifted_v(self, 
//        bm: np.ndarray,
//        is_differentiable=False):
//        """
//        return random vector field:  (B^i(t)V_i(x), i=1,..,d)
//        """
//        def vec1(x: np.ndarray):
//            """
//            x(0)=x1, x(1)=x2, x(2)=e11, x(3)=e12,x(4)=e21,x(5)=e22
//            """
//            v0 = x(2) * _a * x(0) ** _beta * x(1)
//            v1 = x(2) * _b * _rho * x(1) + x(4) * _b * (1. - _rho ** 2.) ** 0.5 * x(1)
//            n0 = x(2) * x(2) * _gamma(0,0,0) + x(2) * x(4) * _gamma(0,1,0) + x(4) * x(2) * _gamma(1,0,0) + x(4) * x(4) * _gamma(1,1,0)
//            n1 = x(2) * x(2) * _gamma(0,0,1) + x(2) * x(4) * _gamma(0,1,1) + x(4) * x(2) * _gamma(1,0,1) + x(4) * x(4) * _gamma(1,1,1)
//            n2 = x(2) * x(3) * _gamma(0,0,0) + x(2) * x(5) * _gamma(0,1,0) + x(4) * x(3) * _gamma(1,0,0) + x(4) * x(5) * _gamma(1,1,0)
//            n3 = x(2) * x(3) * _gamma(0,0,1) + x(2) * x(5) * _gamma(0,1,1) + x(4) * x(3) * _gamma(1,0,1) + x(4) * x(5) * _gamma(1,1,1)
//            if is_differentiable:
//                return tf.convert_to_tensor((v0, v1, n0, n1, n2, n3), np.float32) * bm(0)
//            else:
//                return np.array((v0, v1, n0, n1, n2, n3)) * bm(0)
// 
//        def vec2(x: np.ndarray):
//            """
//            x(0)=x1, x(1)=x2, x(2)=e11, x(3)=e12,x(4)=e21,x(5)=e22
//            """
//            v0 = x(3) * _a * x(0) ** _beta * x(1)
//            v1 = x(3) * _b * _rho * x(1) + x(5) * _b * (1. - _rho ** 2.) ** 0.5 * x(1)
//            n0 = x(3) * x(2) * _gamma(0,0,0) + x(3) * x(4) * _gamma(0,1,0) + x(5) * x(2) * _gamma(1,0,0) + x(5) * x(4) * _gamma(1,1,0)
//            n1 = x(3) * x(2) * _gamma(0,0,1) + x(3) * x(4) * _gamma(0,1,1) + x(5) * x(2) * _gamma(1,0,1) + x(5) * x(4) * _gamma(1,1,1)
//            n2 = x(3) * x(3) * _gamma(0,0,0) + x(3) * x(5) * _gamma(0,1,0) + x(5) * x(3) * _gamma(1,0,0) + x(5) * x(5) * _gamma(1,1,0)
//            n3 = x(3) * x(3) * _gamma(0,0,1) + x(3) * x(5) * _gamma(0,1,1) + x(5) * x(3) * _gamma(1,0,1) + x(5) * x(5) * _gamma(1,1,1)
//            if is_differentiable:
//                return tf.convert_to_tensor((v0, v1, n0, n1, n2, n3), np.float32) * bm(1)
//            else:
//                return np.array((v0, v1, n0, n1, n2, n3)) * bm(1)
//        return (vec1, vec2)
//
//    def get_v0(self):
//        def vec0(x: np.ndarray):
//            v0 = -0.5 * (_a ** 2. * _beta * x(1)**2 * x(0) **  (2. * _beta -1.) + _a * _b * _rho * x(1) * x(0) ** _beta)
//            v1 = -0.5 * _b ** 2. * x(1)
//            return np.array((v0, v1))
//        return vec0
//        
//
//