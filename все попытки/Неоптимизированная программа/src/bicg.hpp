#pragma once
#include "sparsematrix.hpp"
#include "vector.hpp"
#include "preconditioner.hpp"

void BICGStab_precond(
    const SparseMatrix& matrix,
    const Vector& b,
    Vector& x,
    const Preconditioner& preconditioner,
    const unsigned int max_iter,
    const float tolerance);
    
