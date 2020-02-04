#pragma once
#include "sde.h"
namespace sde
{

class Sabr;
class Heston;

template <typename Derived, std::size_t Size>
class VectorField {
public:

    virtual ~VectorField() = default;
    virtual std::unique_ptr<VectorField<Derived, Size>> clone() const = 0;

    template <typename T>
    Eigen::Matrix<T, Size, Size> calcV(const sde::vector_type<T, Size>& x) const {
        return dynamic_cast<const Derived*>(this)->calcV(x);
    }

    template<typename T>
    sde::Tensor<T, Size> calcGDiff(const sde::vector_type<T, Size>& x) const {
        return dynamic_cast<const Derived*>(this)->calcGDiff(x);
    }

    template<typename T>
    sde::vector_type<T, Size> calcV0(const sde::vector_type<T, Size>& x) const {
        return dynamic_cast<const Derived*>(this)->calcV0(x);
    }

    template <typename T>
    Eigen::Matrix<T, 2, 2> calcGInv(const sde::vector_type<T, 2>& x) const
    {
        auto&& v = calcV(x);
        auto&& trans = v.transpose();
        auto&& mat  = v * trans;
        return mat;
    }

    template <typename T>
    sde::Tensor<T, Size> calcGamma(const sde::vector_type<T, Size>& x) const 
    {
        //std::cout << "X" << std::endl;
        //std::cout << x << std::endl;
        auto gInv = this->calcGInv(x);
        auto gDiff = this->calcGDiff(x);
 
        auto&& calc = [&gInv, &gDiff](int i, int j, int k){
            T g = 0;
            for (int m = 0; m < Size; ++m) {
                g = g + 0.5 * gInv(k,m) * (gDiff(i, m, j) + gDiff(m,j,i) - gDiff(i,j,m));
            }
            return g;
        };

        sde::Tensor<T, Size> gamma;
        for (int i = 0; i < Size; ++i) {
            for (int j = 0; j < Size; ++j) {
                for (int k = 0; k < Size; ++k) {
                    gamma(i, j, k) = calc(i, j, k);
                }
            }
        }
        return gamma;
    }

    template <typename T>    
    std::vector<sde::func_ptr_type<sde::lifted_type<T, Size>>>
    getLiftedV(const sde::vector_type<double, Size>& bm) const
    {
        auto&& generateVecField = [this, &bm](int alpha){
            auto&& vec = [this, &bm, alpha](const sde::lifted_type<T, Size>& x){
                //u(0)=x1=x(0), u(1)=xSize=x(1), e(0,0)=e11=x(Size), e(0,1)=e1Size=x(3), e(1,0)=eSize1=x(4), e(1,1)=eSizeSize=x(5)
                sde::vector_type<T, Size> u;
                for (int i = 0; i < Size; ++i) {
                    u(i) = x(i);
                }
                Eigen::Matrix<T, Size, Size> e;
                for (int i = 0; i < Size; ++i) {
                    for (int j = 0; j < Size; ++j) {
                        e(i, j) = x(Size + Size * i + j);
                    }
                }
                
                sde::Tensor<T, Size> gamma = calcGamma(u);
                auto&& calcV = [&e, &gamma, alpha](int i, int j){
                    T ret = 0;
                    for (int k = 0; k < Size; ++k) {
                        for (int l = 0; l < Size; ++l) {
                            ret = ret + gamma(k, l, i) * e(l, j) * e(k, alpha); 
                        }
                    }
                    return ret;
                };

                sde::lifted_type<T, Size> v;
                for (int i = 0; i < Size; ++i) {
                    v(i) = e(i, 0);
                }
                for (int i = 0; i < Size; ++i) {
                    for (int j = 0; j < Size; ++j) {
                        v(Size + i * Size + j) = -calcV(i, j);
                    }
                }
                //v << v0, v1, v00, v01, v10, v11;
                //std::cout << "v = " << v << std::endl;
                sde::lifted_type<T, Size> w = bm(alpha) * v;
                //return x;
                return w;
            };
            return vec;
        };
        
        return std::vector<sde::func_ptr_type<sde::lifted_type<T, Size>>>{
            std::make_shared<sde::function_type<sde::lifted_type<T, Size>>>(generateVecField(0)),
            std::make_shared<sde::function_type<sde::lifted_type<T, Size>>>(generateVecField(1))
        };
    }

    template <typename T>
    sde::function_type<sde::vector_type<T, Size>> getV0() const 
    {
        auto&& vec0 = [this](const sde::vector_type<T, Size>& x) {
            auto&& gInv = calcGInv(x);
            auto&& gamma = calcGamma(x);
            sde::vector_type<T, Size> v;
            
            for (int i = 0; i < Size; ++i) {
                for (int k = 0; k < Size; ++k) {
                    for (int l = 0; l < Size; ++l) {
                        v(i) = v(i) + 0.5 * gInv(l, k) * gamma(k, l, i);
                    }
                }
            }
            return v;
        };
        return vec0;
    }
};
   
} // namespace sde