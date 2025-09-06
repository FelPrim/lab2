#include "sparsematrix.hpp"
extern "C"{
    #include <matio.h>
}
#include <string_view>
#include <cassert>
#include <vector>
#include <cstring>
#include <cstdlib>


char *convert_string_view_to_c(std::string_view str){
    char *cstr = std::malloc(str.size() + 1);
    str::memcpy<char>(cstr, str.data(), str.size());
    cstr[str.size()] = '\0';
    return cstr;
}

constexpr Vector SparseMatrix::apply(const Vector& vector) const {
    assert(this->num_cols == vector.size());
    Vector result(this->num_rows);

    /*
    ( 1  2  0  0  0) (1) = ( 1*1 + 0*2 + 2*0 + 3*0 + 0*0 )
    ( 0  0  3  0  4) (0) = ( 1*0 + 0*0 + 2*3 + 3*0 + 0*4 )
    ( 0  0  0  0  0) (2) = ( 1*0 + 0*0 + 2*0 + 3*0 + 0*0 )
    ( 0  0  5  0  0) (3) = ( 1*0 + 0*0 + 2*5 + 3*0 + 0*0 )
    ( 6  7  8  0  0) (0) = ( 1*6 + 0*7 + 2*8 + 3*0 + 0*0 )
    CSR:
    data    = ( 1 2 3 4 5 6 7 8)
    indices = ( 0 1 2 4 2 0 1 2)
    indptr  = ( 0 2 4 4 5 8)
    CSC:
    data    = ( 1 6 2 7 3 5 8 4)
    jndices = ( 0 4 0 4 1 3 4 1)
    jndptr  = ( 0 2 4 7 7 8)


    0 способ умножения (просто умножение матрицы на вектор):
    c[i] += A[i][j]*b[j]

    1 способ умножения:
    for i:
        for j in range(indptr[i], indptr[i+1]): // csr
            c[i] += data[j]*vector[indices[j]]

    2 способ:
    for j:
        for i in range(jndptr[j], jndptr[j+1]): //csc
            c[jndices[i]] += data[i]*vector[j]
    Наверное, лучше, если скачки будут в vector, а не c
    */
    for (unsigned int i = 0; i < this->num_rows; ++i){
        for (unsigned int j = indptr[i]; j<indptr[i+1]; ++j){
            result[i] += this->data[j]*vector[this->indices[j]];
        }
    }
    return result;
}

void load(
    std::string_view filepath,
    std::string_view varname,
    std::string_view fieldname){
    
    char *cfilepath = convert_string_view_to_c(filepath);
    char *cvarname = convert_string_view_to_c(varname);
    char *cfieldname = convert_string_view_to_c(fieldname);

    mat_t *file = Mat_Open(cfilepath, MAT_ACC_RDONLY);
    if (!file)
        goto NOFILE;

    matvar_t *var = Mat_VarRead(file, cvarname);
    Mat_Close(file);
    if (!var)
        goto NOVAR;
    if (var->class_type != MAT_C_STRUCT)
        goto NOVAR;
    matvar_t *field = Mat_VarGetStructFieldByName(var, cfieldname, 0);
    if (!field)
        goto NOFIELD;
    if (field->class_type != MAT_C_SPARSE || field->data_type != MAT_T_DOUBLE || field->data == NULL)
        goto NOFIELD;
    
    mat_sparse_t *input = field->data;

    
NOFIELD:
    std::free(cfieldname);
NOVAR:
    std::free(cvarname);
NOFILE:
    std::free(cfilepath);
}
