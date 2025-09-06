#include "bicg.hpp"
#include <cmath>

#include "useful_math.hpp"

void BICGStab_precond(
    const SparseMatrix& matrix,
    const Vector& b,
    Vector& x,
    const Preconditioner& preconditioner,
    const unsigned int max_iter,
    const float tolerance){
    
    // *x = 0;
    Vector r = b;
    Vector r0_hat = b;
    double rho_prev = 1;
    double alpha = 1;
    double omega = 1;
    Vector p = b;
    Vector v = 0;
    Vector p_hat{};
    Vector s{};
    Vector t{};
    double norm_b = b.norm();

    for (unsigned int i = 0; i < max_iter; ++i){
        double rho = dot(r0_hat, r);
        if (std::abs(rho) < eps)
            break;

        if (i != 0){
            double beta = (rho / rho_prev) * alpha / omega;
            p = r + beta * (p - omega * v);
        }

        p_hat = preconditioner.solve(p);
        v = matrix.apply(p_hat);
        alpha = rho / dot(r_hat, v);

        s = r - alpha*v;
        if (s.norm() < tol){
            x = x + alpha * p_hat;
            break;
        }
        
        s_hat = preconditioner.solve(s);
        t = A * s_hat;
        omega = dot(t, s) / dot(t, t);

        x = x + alpha * p_hat + omega * s_hat;
        r = s - omega * t;

        if (r.norm() < tol)
            break;

        rho_prev = rho;
    }
}
