#pragma once
#include "useful_math.hpp"
#include <vector>
#include <string_view>
#include "vector.hpp"


// CSR
class SparseMatrix{
public:
    SparseMatrix() = default;
    explicit SparseMatrix(
        const std::vector<numeric>& data,
        const std::vector<unsigned int>& indices,
        const std::vector<unsigned int>& indptr,
        unsigned int num_rows,
        unsigned int num_cols):
        data(data), indices(indices), indptr(indptr),
        num_rows(num_rows), num_cols(num_cols){}

    constexpr Vector apply(const Vector&) const;
    void load(
        std:string_view filepath,
        std::string_view varname,
        std::string_view fieldname);

private:
    std::vector<numeric> data;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> indptr;
    unsigned int num_rows;
    unsigned int num_cols;
};
