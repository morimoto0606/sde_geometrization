#pragma once
#include <iostream>
#include <memory>
#include <Eigen/Core>
#include "sde.h"

namespace sde
{
template <typename T>
std::size_t size(const T&x) {
    return std::size(x);
}
template <>
std::size_t size(const double& x) {
    return 1;
}


template <typename T, std::size_t Size>
class RungeKutta5 {
public:
    using value_type = sde::vector_type<T, Size>;
    using func_type = sde::function_type<T, Size>;
    using func_ptr_type = sde::func_ptr_type<T, Size>;

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

private:
        Eigen::Matrix<T, 6, 6> _a;// = Eigen::MatrixXd::Zero(6,6);
        Eigen::Matrix<T, 6, 1> _b;// = Eigen::MatrixXd::Zero(6,1);

public:
    value_type solve(
        const T& h,
        const func_type& vecfield,
        const value_type& ini_val) const 
    {
        Eigen::Matrix<T, Size, 6> k;
        k.col(0) = vecfield(ini_val);
        k.col(1) = vecfield(ini_val + h * (_a(1,0) * k.col(0)));
        k.col(2) = vecfield(ini_val + h * (_a(2,0) * k.col(0) + _a(2,1) * k.col(1)));
        k.col(3) = vecfield(ini_val + h * (_a(3,0) * k.col(0) + _a(3,1) * k.col(1) + _a(3,2) * k.col(2)));
        k.col(4) = vecfield(ini_val + h * (_a(4,0) * k.col(0) + _a(4,1) * k.col(1) + _a(4,2) * k.col(2) + _a(4,3) * k.col(3)));
        k.col(5) = vecfield(ini_val + h * (_a(5,0) * k.col(0) + _a(5,1) * k.col(1) + _a(5,2) * k.col(2) + _a(5,3) * k.col(3) + _a(5,4) * k.col(4)));
        auto ret = ini_val + h * k * _b;
        return ret;
    }
    
    value_type solveIterative(
        const T& h,
        const std::vector<func_ptr_type>& vecfields,
        const value_type& ini_val) const 
    {
        auto x = ini_val;
        for (auto v: vecfields) {
            x = this->solve(h, *v, x);
        }
        return x;
    }
};
}
    



//class RungeKutta5Differentiable:
//    def __init__(self):
//        self._a = np.zeros((6, 6))
//        self._a[1,0] = 2./5
//        self._a[2,0] = 11./64
//        self._a[2,1] = 5./64
//        self._a[3,2] = 0.5
//        self._a[4,0] = 3./64
//        self._a[4,1] = -15./64
//        self._a[4,2] = 3./8
//        self._a[4,3] = 9./16
//        self._a[5,1] = 5./7
//        self._a[5,2] = 6./7
//        self._a[5,3] = -12./7
//        self._a[5,4] = 8./7
//
//        self._b=tf.convert_to_tensor(np.array([7.,0.,32.,12.,32.,7.])/90, np.float32)
//
//    def solve(self,
//        h: float,
//        vecfield: Callable[[np.ndarray], np.ndarray],
//        ini_val: np.ndarray):
//        k0 = vecfield(ini_val)
//        k1 = vecfield(ini_val + h * (self._a[1,0] * k0))
//        k2 = vecfield(ini_val + h * (self._a[2,0] * k0 + self._a[2,1] * k1))
//        k3 = vecfield(ini_val + h * (self._a[3,0] * k0 + self._a[3,1] * k1 + self._a[3,2] * k2))
//        k4 = vecfield(ini_val + h * (self._a[4,0] * k0 + self._a[4,1] * k1 + self._a[4,2] * k2 + self._a[4,3] * k3))
//        k5 = vecfield(ini_val + h * (self._a[5,0] * k0 + self._a[5,1] * k1 + self._a[5,2] * k2 + self._a[5,3] * k3 + self._a[5,4] * k4))
//        k = tf.convert_to_tensor([k0, k1, k2, k3, k4, k5], np.float32)
//        ret = ini_val + h * tf.tensordot(self._b, k, 1)
//        return ret
//
//    def solve_iterative(self,
//        h: float,
//        vecfields: List[Callable[[np.ndarray], np.ndarray]],
//        ini_val: np.ndarray):
//        h = tf.convert_to_tensor(h, np.float32)
//        x = ini_val
//        for v in vecfields:
//            x = self.solve(h, v, x)
//        return x
//
//class RungeKutta4:
//    def __init__(self):
//        self._a = np.zeros((4, 4))
//        self._a[1,0] = 0.5
//        self._a[2,1] = 0.5
//        self._a[3,2] = 1.0
//        self._b=np.array([1.,2.,2.,1.])/6
//
//    def solve(self,
//        h: float,
//        vecfield: Callable[[np.ndarray], np.ndarray],
//        ini_val: np.ndarray):
//        k0 = vecfield(ini_val)
//        k1 = vecfield(ini_val + h * (self._a[1,0] * k0))
//        k2 = vecfield(ini_val + h * (self._a[2,0] * k0 + self._a[2,1] * k1))
//        k3 = vecfield(ini_val + h * (self._a[3,0] * k0 + self._a[3,1] * k1 + self._a[3,2] * k2))
//        k = np.array([k0, k1, k2, k3])
//        ret = ini_val + h * np.dot(self._b, k)
//        return ret
//
//    def solve_iterative(self,
//        h: float,
//        vecfields: List[Callable[[np.ndarray], np.ndarray]],
//        ini_val: np.ndarray):
//        x = ini_val
//        for v in vecfields:
//            x = self.solve(h, v, x)
//        return x   
//} // namespace sde
//