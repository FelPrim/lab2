#include "math_stuff.h"


// Matrix[X][Y] * Vector[X] = Vector[Y]
void bicgstab(const SparseMatrix * restrict matrix,
    Vector_X * restrict x,
    const Vector_Y * restrict b,
    const Preconditioner * restrict  preconditioner,
    const unsigned int max_iter,
    const double tolerance);

// matrix[X][Y] * x[X] = y[Y]
void matrix_apply(const SparseMatrix * restrict matrix,
    const VectorX * restrict x,
    VectorY * restrict y);

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

