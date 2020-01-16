#pragma once
#include <memory>
#include <Eigen/Core>
#include <autodiff/reverse.hpp>
#include <autodiff/forward.hpp>
#include <autodiff/reverse/eigen.hpp>




namespace sde {
    template <typename T, std::size_t Size>
    using vector_type = Eigen::Matrix<T, Size, 1>;

    template <typename T, std::size_t Size>
    using function_type = std::function<vector_type<T, Size>(const vector_type<T, Size>&)>;

    template <typename T, std::size_t Size>
    using func_ptr_type = std::shared_ptr<const function_type<T, Size>>;

}