#include "lowlevelmath.h"
#include "memory_stuff.h"
#include <stdlib.h>

// matrix[X][Y] * x[X] = y[Y]
void matrix_apply(const SparseMatrix * const restrict matrix,
    const VectorX * const restrict x,
    VectorY * const restrict y);

inline void _matrix_apply(const aligned_double * const restrict A,
    const aligned_double * const restrict x,
    aligned_double * const restrict b){
    

}

// preconditioner[Y][X] * x[Y] = y[X]
void preconditioner_apply(const Preconditioner * restrict preconditioner,
    const VectorY * restrict x,
    VectorX * restrict y);

// Vector[Y] - Vector[Y] = Vector[Y]
void vectory_difference(const VectorY * restrict left,
    const VectorY * restrict right,
    VectorY * restrict result);

// double * Vector[Y] = Vector[Y]
void vectory_multiply(const double number,
    const VectorY * restrict vec,
    VectorY * restrict result);

// double * Vector[X] = Vector[X]
void vectorx_multiply(const double number,
    const VectorX * restrict vec,
    VectorX * restrict result);

// Vector[Y] + Vector[Y] = Vector[Y]
void vectory_sum(const VctorY * restrict left,
    const VectorY * restrict right,
    VectorY * restrict result);

// Vector[X] + Vector[X] = Vector[X]
void vectorx_sum(const VectorX * restrict left,
    const VectorX * restrict right,
    VectorX * restrict result);

// Vector[Y] * Vector[Y]
double dot_product(const VectorY * restrict left, const VectorY * restrict right);

// ||Vector[Y]||
double norm(const VectorY * restrict v);

