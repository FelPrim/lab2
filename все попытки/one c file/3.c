#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <matio.h>
#include <assert.h>

#define ALIGN 32

static void* xaligned_malloc(size_t nbytes){
    void *p = NULL;
#if defined(_WIN32)
    p = _aligned_malloc(nbytes, ALIGN);
#define xaligned_free _aligned_free
#else
    if(posix_memalign(&p, ALIGN, nbytes)) p = NULL;
#define xaligned_free free
#endif
    if (!p){ fprintf(stderr, "OOM for %zu bytes\n", nbytes); exit(1);} 
    return p;
}

typedef struct Vector{
    alignas(32) double *ptr;
    uint32_t size;
} Vector;

typedef struct CSRMatrix{
    alignas(32) double *data;
    alignas(32) uint32_t *indices;
    uint32_t *indptr; // length rows+1
    uint32_t rows;
    uint32_t nnz;
} CSRMatrix;

typedef struct {
    CSRMatrix *blocks;    /* array of CSR blocks */
    uint32_t *ranges;     /* block start indices, length nb+1 */
    uint32_t nb;          /* number of blocks */
} BlockJacobi;

static BlockJacobi GBlockPrecond = { NULL, NULL, 0 };

//////////////////////////////////////////////
static double TOLERANCE = 1e-9;
static unsigned int MAXITER = 10000;
static unsigned int THREADS_NUM = 12;
static unsigned int BIGCACHE_SZ = 512000;
static unsigned int BW = 64;              
unsigned int STRIPE_ROWS = 0;
/////////////////////////////////////////////
static inline void calculate_stripe_rows(unsigned int bw){
    STRIPE_ROWS = BIGCACHE_SZ/(16*bw) - 1;
}

static inline double wall_seconds(){ return omp_get_wtime(); }



static void csr_free(CSRMatrix *A){
    if(!A) return;
    xaligned_free(A->data); 
    xaligned_free(A->indices); 
    xaligned_free(A->indptr);
    memset(A,0,sizeof(*A));
}

static CSRMatrix csr_empty(uint32_t n, uint32_t nnz){
    CSRMatrix A; memset(&A,0,sizeof(A));
    A.rows = n; A.nnz = nnz;
    A.data    = (double*)xaligned_malloc(sizeof(double)*nnz);
    A.indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*nnz);
    A.indptr  = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(n+1));
    return A;
}

static Vector vec_empty(uint32_t n){
    Vector v; 
    v.size=n; 
    v.ptr=(double*)xaligned_malloc(sizeof(double)*n);
    return v;
}

static void vec_free(Vector *v){
    if(!v) return; 
    xaligned_free(v->ptr); 
    v->ptr=NULL; 
    v->size=0; 
}

static inline void vzero(double* restrict y, uint32_t n){
    memset(y,0,sizeof(double)*n);
}
static inline void vsum(const double* restrict x, const double* restrict y, double * restrict z, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < n; ++i) 
        z[i] = x[i]+y[i];
}
static inline void vdif(const double* restrict x, const double* restrict y, double * restrict z, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < n; ++i) 
        z[i] = x[i]-y[i];
}
static inline void vaxpy(double* restrict y, double a, const double* restrict x, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < n; ++i) 
        y[i] += a*x[i];
}
static inline void vxpay(double* restrict y, double b, const double* restrict x, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < n; ++i) 
        y[i] = x[i] + b*y[i];
}
static inline void vscale(double* restrict y, double a, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) y[i] *= a;
}
static inline double vdot(const double* restrict a, const double* restrict b, uint32_t n){
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for(uint32_t i=0;i<n;++i) sum += a[i]*b[i];
    return sum;
}
static inline double vnorm2(const double* restrict a, uint32_t n){ return sqrt(vdot(a,a,n)); }

///////////////////////////////////////////
static bool read_binary_vector(const char* path, Vector* out){
    FILE* f = fopen(path, "rb"); if(!f) return false;
    uint32_t n=0; if(fread(&n,sizeof(uint32_t),1,f)!=1){ fclose(f); return false; }
    *out = vec_empty(n);
    if(fread(out->ptr,sizeof(double),n,f)!=(size_t)n){ fclose(f); return false; }
    fclose(f); return true;
}

static void write_binary_vector(const char* path, const Vector* v){
    FILE* f = fopen(path, "wb"); if(!f){ perror("fopen"); exit(1);} 
    fwrite(&v->size,sizeof(uint32_t),1,f);
    fwrite(v->ptr,sizeof(double),v->size,f);
    fclose(f);
}

static Vector make_or_load_b(const char* path, uint32_t n){
    Vector b; if(read_binary_vector(path,&b)) return b;
    b = vec_empty(n);
    srand(42u);
    for(uint32_t i=0;i<n;++i){
        b.ptr[i] = (double)rand() / (double)RAND_MAX - 0.5;
    }
    write_binary_vector(path,&b);
    return b;
}

////////////////////////////////////////////////////////////////
static void csc_to_csr_and_split(const mat_sparse_t* A, uint32_t N, unsigned int bw,
                                 CSRMatrix* Band, CSRMatrix* Other){

    const uint32_t* jc = (const uint32_t*)A->jc;
    const uint32_t* ir = (const uint32_t*)A->ir;
    const double*  pr = (const double*) A->data;
    const uint32_t nnz = (uint32_t)A->ndata;

    uint32_t *row_band = (uint32_t*)calloc(N,sizeof(uint32_t));
    uint32_t *row_other= (uint32_t*)calloc(N,sizeof(uint32_t));
    for(uint32_t col=0; col<N; ++col){
        for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
            uint32_t r = ir[k];
            if((int)abs((int)r-(int)col) <= (int)bw) row_band[r]++; else row_other[r]++;
        }
    }
    
    memset(Band, 0, sizeof(CSRMatrix));
    memset(Other, 0, sizeof(CSRMatrix));
    Band->rows = N; Other->rows = N;
    Band->indptr  = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(N+1));
    Other->indptr = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(N+1));
    Band->indptr[0]=0; Other->indptr[0]=0;
    for(uint32_t i=0;i<N;++i){
        Band->indptr[i+1]  = Band->indptr[i]  + row_band[i];
        Other->indptr[i+1] = Other->indptr[i] + row_other[i];
    }
    Band->nnz  = Band->indptr[N];
    Other->nnz = Other->indptr[N];
    Band->data    = (double*)xaligned_malloc(sizeof(double)*Band->nnz);
    Band->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*Band->nnz);
    Other->data    = (double*)xaligned_malloc(sizeof(double)*Other->nnz);
    Other->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*Other->nnz);

    uint32_t *wb = (uint32_t*)calloc(N,sizeof(uint32_t));
    uint32_t *wo = (uint32_t*)calloc(N,sizeof(uint32_t));
    

    for(uint32_t col=0; col<N; ++col){
        for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
            uint32_t r = ir[k];
            if((int)abs((int)r-(int)col) <= (int)bw){
                uint32_t pos = Band->indptr[r] + wb[r]++;
                Band->indices[pos] = col; Band->data[pos] = pr[k];
            }else{
                uint32_t pos = Other->indptr[r] + wo[r]++;
                Other->indices[pos] = col; Other->data[pos] = pr[k];
            }
        }
    }
    free(wb); free(wo); free(row_band); free(row_other);

}
/* helper: comparator for qsort (descending by absolute value) */
typedef struct { double val; int idx; } paird;

/* ---------- Block-Jacobi SPAI (block inverse on diagonal) ----------
   Replaces previous spai_build_improved. Produces CSRMatrix M which
   approximates A^{-1} by inverse of diagonal blocks of size bs.
*/
static int invert_dense_block_gj(double *A /* n x n row-major */, int n, double reg_eps){
    /* Gauss-Jordan with partial pivoting. A overwritten by inverse on success. */
    int i,j,k;
    double *aug = (double*)malloc(sizeof(double)* (size_t)n * (size_t)(2*n));
    if(!aug) return 1;
    /* build augmented [A | I] */
    for(i=0;i<n;++i){
        for(j=0;j<n;++j) aug[(size_t)i*(2*n) + j] = A[(size_t)i*(size_t)n + j];
        for(j=0;j<n;++j) aug[(size_t)i*(2*n) + (n + j)] = (i==j) ? 1.0 : 0.0;
    }
    for(k=0;k<n;++k){
        /* partial pivot: find max abs in column k, rows k..n-1 */
        double maxv = 0.0; int piv = -1;
        for(i=k;i<n;++i){
            double v = fabs(aug[(size_t)i*(2*n) + k]);
            if(v > maxv){ maxv = v; piv = i; }
        }
        if(maxv < 1e-18){
            /* tiny pivot -> try regularize diagonal and continue */
            if(reg_eps <= 0.0){ free(aug); return 1; }
            aug[(size_t)k*(2*n) + k] += reg_eps;
            maxv = fabs(aug[(size_t)k*(2*n) + k]);
            if(maxv < 1e-18){ free(aug); return 1; }
        }
        if(piv != k){
            /* swap rows k and piv */
            for(j=0;j<2*n;++j){
                double t = aug[(size_t)k*(2*n) + j];
                aug[(size_t)k*(2*n) + j] = aug[(size_t)piv*(2*n) + j];
                aug[(size_t)piv*(2*n) + j] = t;
            }
        }
        /* normalize row k */
        double diag = aug[(size_t)k*(2*n) + k];
        for(j=0;j<2*n;++j) aug[(size_t)k*(2*n) + j] /= diag;
        /* eliminate other rows */
        for(i=0;i<n;++i){
            if(i==k) continue;
            double fac = aug[(size_t)i*(2*n) + k];
            if(fac == 0.0) continue;
            for(j=0;j<2*n;++j) aug[(size_t)i*(2*n) + j] -= fac * aug[(size_t)k*(2*n) + j];
        }
    }
    /* copy inverse back to A (right half) */
    for(i=0;i<n;++i){
        for(j=0;j<n;++j){
            A[(size_t)i*(size_t)n + j] = aug[(size_t)i*(2*n) + (n + j)];
        }
    }
    free(aug);
    return 0;
}

static void csr_apply(const CSRMatrix* A, const double* restrict x, double* restrict y){
    const uint32_t n = A->rows; 
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i){
        double sum=0.0;
        for(uint32_t k=A->indptr[i]; k<A->indptr[i+1]; ++k){ sum += A->data[k]*x[A->indices[k]]; }
        y[i]=sum;
    }
}


static void spai_build_blockdiag(const mat_sparse_t* A, uint32_t N, CSRMatrix* M){
    const uint32_t* jc = (const uint32_t*)A->jc;
    const uint32_t* ir = (const uint32_t*)A->ir;
    const double* pr = (const double*)A->data;

    const uint32_t bs = 16;           /* block size: try 8,16,32 */
    const double reg_scale = 1e-8;    /* regularization added to block diagonal if needed */
    const double keep_tol = 1e-12;    /* threshold for small values (we keep full block by default) */

    /* allocate diag for fallback */
    double *diag = (double*)malloc(sizeof(double)*N);
    if(!diag){ fprintf(stderr,"OOM diag\n"); exit(1); }
    for(uint32_t i=0;i<N;++i) diag[i] = 0.0;
    for(uint32_t col=0; col<N; ++col){
        for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
            uint32_t r = ir[k];
            if(r==col){ diag[col] = pr[k]; break; }
        }
    }
    for(uint32_t i=0;i<N;++i) if(fabs(diag[i]) < 1e-300) diag[i] = 1.0;

    /* compute row_count: each row within a block will have up to block_size nonzeros (dense block inverse) */
    uint32_t nb = (N + bs - 1) / bs;
    uint32_t *row_count = (uint32_t*)calloc((size_t)N, sizeof(uint32_t));
    if(!row_count){ fprintf(stderr,"OOM row_count\n"); exit(1); }
    for(uint32_t b=0;b<nb;++b){
        uint32_t r0 = b * bs;
        uint32_t s = (r0 + bs <= N) ? bs : (N - r0);
        for(uint32_t i=0;i<s;++i) row_count[r0 + i] = s;
    }

    /* build indptr */
    M->rows = N;
    M->indptr = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(N+1));
    if(!M->indptr){ fprintf(stderr,"OOM indptr\n"); exit(1); }
    M->indptr[0] = 0;
    for(uint32_t i=0;i<N;++i) M->indptr[i+1] = M->indptr[i] + row_count[i];
    M->nnz = M->indptr[N];
    M->data = (double*)xaligned_malloc(sizeof(double)*(size_t)M->nnz);
    M->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(size_t)M->nnz);
    if((!M->data) || (!M->indices)){ fprintf(stderr,"OOM data/indices\n"); exit(1); }

    /* second pass: for each block build dense submatrix and invert */
    uint32_t pos_write = 0;
    for(uint32_t b=0;b<nb;++b){
        uint32_t r0 = b * bs;
        uint32_t s = (r0 + bs <= N) ? bs : (N - r0);
        /* allocate dense s x s (row-major) filled with zeros */
        double *B = (double*)calloc((size_t)s * (size_t)s, sizeof(double));
        if(!B){ fprintf(stderr,"OOM block B\n"); exit(1); }

        /* fill B from A: scan columns r0..r0+s-1 */
        for(uint32_t col = r0; col < r0 + s; ++col){
            uint32_t p = col - r0;
            for(uint32_t k = jc[col]; k < jc[col+1]; ++k){
                uint32_t row = ir[k];
                if(row >= r0 && row < r0 + s){
                    uint32_t i = row - r0;
                    B[(size_t)i*(size_t)s + p] = pr[k];
                }
            }
        }

        /* regularize small diagonal if singular-ish */
        double maxdiag = 0.0;
        for(uint32_t i=0;i<s;++i) maxdiag = fmax(maxdiag, fabs(B[(size_t)i*(size_t)s + i]));
        double reg_eps = reg_scale * fmax(1.0, maxdiag);
        /* attempt inversion */
        int ok = invert_dense_block_gj(B, (int)s, reg_eps);
        if(ok != 0){
            /* if failed, try stronger regularization and retry */
            reg_eps *= 1e3;
            for(uint32_t i=0;i<s;++i) B[(size_t)i*(size_t)s + i] += reg_eps;
            ok = invert_dense_block_gj(B, (int)s, reg_eps);
        }
        /* if still failed, fallback to diagonal inverse for block */
        if(ok != 0){
            for(uint32_t i=0;i<s;++i){
                uint32_t global_r = r0 + i;
                uint32_t idx = M->indptr[global_r] + (global_r - r0); /* place at relative position */
                M->indices[idx] = global_r;
                M->data[idx] = 1.0 / diag[global_r];
            }
            free(B);
            pos_write += s * s;
            continue;
        }

        /* write inverse block B (which now contains inverse) into CSR row-by-row */
        for(uint32_t i=0;i<s;++i){
            uint32_t global_r = r0 + i;
            uint32_t row_start = M->indptr[global_r];
            /* we will write exactly s elements in this row: columns r0..r0+s-1 */
            for(uint32_t p=0;p<s;++p){
                uint32_t idx = row_start + p;
                uint32_t global_c = r0 + p;
                double val = B[(size_t)i*(size_t)s + p];
                if(fabs(val) < keep_tol) val = 0.0; /* tiny clipping */
                M->indices[idx] = global_c;
                M->data[idx] = val;
            }
        }
        free(B);
        pos_write += s * s;
    }

    free(row_count);
    free(diag);
}

/* ---------------- extract block A[r0..r0+s-1, r0..r0+s-1] into CSR B ---------------- */
static void csr_extract_block(const mat_sparse_t* A_src, uint32_t r0, uint32_t s, CSRMatrix *B){
    const uint32_t *jc = (const uint32_t*)A_src->jc;
    const uint32_t *ir = (const uint32_t*)A_src->ir;
    const double *pr = (const double*)A_src->data;

    /* first, count nnz per row inside the block */
    uint32_t *row_counts = (uint32_t*)calloc((size_t)s, sizeof(uint32_t));
    if(!row_counts){ fprintf(stderr,"OOM row_counts\n"); exit(1); }

    uint32_t nnz = 0;
    for(uint32_t col = r0; col < r0 + s; ++col){
        for(uint32_t k = jc[col]; k < jc[col+1]; ++k){
            uint32_t row = ir[k];
            if(row >= r0 && row < r0 + s){
                row_counts[row - r0] += 1;
                ++nnz;
            }
        }
    }

    B->rows = s;
    B->indptr = (uint32_t*)malloc(sizeof(uint32_t)*(s+1));
    B->indices = (uint32_t*)malloc(sizeof(uint32_t)*nnz);
    B->data = (double*)malloc(sizeof(double)*nnz);
    if(!B->indptr || !B->indices || !B->data){ fprintf(stderr,"OOM block extract\n"); exit(1); }

    B->indptr[0] = 0;
    for(uint32_t i=0;i<s;++i) B->indptr[i+1] = B->indptr[i] + row_counts[i];

    /* fill rows: maintain a cursor per row */
    uint32_t *cursor = (uint32_t*)malloc(sizeof(uint32_t)*s);
    if(!cursor){ fprintf(stderr,"OOM cursor\n"); exit(1); }
    memcpy(cursor, B->indptr, sizeof(uint32_t)*s);

    for(uint32_t col = r0; col < r0 + s; ++col){
        uint32_t col_local = col - r0;
        for(uint32_t k = jc[col]; k < jc[col+1]; ++k){
            uint32_t row = ir[k];
            if(row >= r0 && row < r0 + s){
                uint32_t row_local = row - r0;
                uint32_t idx = cursor[row_local]++;
                B->indices[idx] = col_local;
                B->data[idx] = pr[k];
            }
        }
    }

    free(row_counts);
    free(cursor);
}

/* ---------------- compact BiCGStab for small CSRMatrix blocks ----------------
   solves A x = b, A is CSRMatrix (rows == cols == n). Returns 0 if converged/ok, 1 on failure.
   parameters: maxit, tol.
*/
static int bicgstab_block(const CSRMatrix *A, const double *b_in, double *x_out, int maxit, double tol){
    uint32_t n = A->rows;
    double *r = (double*)malloc(sizeof(double)*n);
    double *r0 = (double*)malloc(sizeof(double)*n);
    double *p = (double*)malloc(sizeof(double)*n);
    double *v = (double*)malloc(sizeof(double)*n);
    double *s = (double*)malloc(sizeof(double)*n);
    double *t = (double*)malloc(sizeof(double)*n);
    double *tmp = (double*)malloc(sizeof(double)*n);
    if(!r || !r0 || !p || !v || !s || !t || !tmp){ fprintf(stderr,"OOM bicg workspace\n"); exit(1); }

    /* initial x = 0 */
    for(uint32_t i=0;i<n;++i) x_out[i] = 0.0;

    csr_apply(A, x_out, tmp); /* tmp = A*x (zero) */
    for(uint32_t i=0;i<n;++i){ r[i] = b_in[i] - tmp[i]; r0[i] = r[i]; p[i]=0.0; v[i]=0.0; }

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double rho = 0.0;
    double normb = 0.0;
    for(uint32_t i=0;i<n;++i) normb += b_in[i]*b_in[i];
    normb = sqrt(normb);
    if(normb == 0.0) normb = 1.0;

    double resid = 0.0;
    /* initial residual norm */
    {
        double rr = 0.0;
        for(uint32_t i=0;i<n;++i) rr += r[i]*r[i];
        resid = sqrt(rr) / normb;
        if(resid < tol){
            /* already solved */
            free(r); free(r0); free(p); free(v); free(s); free(t); free(tmp);
            return 0;
        }
    }

    for(int iter=1; iter<=maxit; ++iter){
        /* rho = (r0, r) */
        rho = 0.0;
        for(uint32_t i=0;i<n;++i) rho += r0[i]*r[i];
        if(fabs(rho) < 1e-30) break;
        double beta = (rho / rho_old) * (alpha / omega);
        for(uint32_t i=0;i<n;++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        /* v = A * p */
        csr_apply(A, p, v);
        /* alpha = rho / (r0, v) */
        double r0v = 0.0;
        for(uint32_t i=0;i<n;++i) r0v += r0[i] * v[i];
        if(fabs(r0v) < 1e-30) break;
        alpha = rho / r0v;
        /* s = r - alpha * v */
        for(uint32_t i=0;i<n;++i) s[i] = r[i] - alpha * v[i];
        /* check s */
        double s_norm = 0.0;
        for(uint32_t i=0;i<n;++i) s_norm += s[i]*s[i];
        s_norm = sqrt(s_norm) / normb;
        if(s_norm < tol){
            /* x += alpha * p; done */
            for(uint32_t i=0;i<n;++i) x_out[i] += alpha * p[i];
            free(r); free(r0); free(p); free(v); free(s); free(t); free(tmp);
            return 0;
        }
        /* t = A * s */
        csr_apply(A, s, t);
        double ts = 0.0, tt = 0.0;
        for(uint32_t i=0;i<n;++i){ ts += t[i] * s[i]; tt += t[i] * t[i]; }
        if(fabs(tt) < 1e-30) break;
        omega = ts / tt;
        /* x += alpha * p + omega * s */
        for(uint32_t i=0;i<n;++i) x_out[i] += alpha * p[i] + omega * s[i];
        /* r = s - omega * t */
        for(uint32_t i=0;i<n;++i) r[i] = s[i] - omega * t[i];
        /* check convergence */
        double rr = 0.0;
        for(uint32_t i=0;i<n;++i) rr += r[i]*r[i];
        resid = sqrt(rr) / normb;
        if(resid < tol){
            free(r); free(r0); free(p); free(v); free(s); free(t); free(tmp);
            return 0;
        }
        if(fabs(omega) < 1e-30) break;
        rho_old = rho;
    }
    /* not converged */
    free(r); free(r0); free(p); free(v); free(s); free(t); free(tmp);
    return 1;
}

/* ---------------- build blocks: extract diagonal blocks of size Bsize ----------------
   blocks_out - will be malloc'ed as array of CSRMatrix[nb]; ranges_out malloc'ed (nb+1)
*/
static void build_block_jacobi_blocks(const mat_sparse_t* A_src, uint32_t N, uint32_t Bsize,
                                      CSRMatrix **blocks_out, uint32_t **ranges_out, uint32_t *nb_out){
    uint32_t nb = (N + Bsize - 1) / Bsize;
    CSRMatrix *blocks = (CSRMatrix*)malloc(sizeof(CSRMatrix)*(size_t)nb);
    uint32_t *ranges = (uint32_t*)malloc(sizeof(uint32_t)*(size_t)(nb+1));
    for(uint32_t b=0;b<nb;++b) ranges[b] = b * Bsize;
    ranges[nb] = N;

    for(uint32_t b=0;b<nb;++b){
        uint32_t r0 = ranges[b];
        uint32_t s = ranges[b+1] - ranges[b];
        csr_extract_block(A_src, r0, s, &blocks[b]);
    }
    *blocks_out = blocks;
    *ranges_out = ranges;
    *nb_out = nb;
}

/* ---------------- apply block-Jacobi preconditioner ----------------
   blocks: array of CSRMatrix[nb], ranges: nb+1, rhs_global is length N, x_global is out length N.
   For each block we solve A_block * x_block = rhs_block using bicgstab_block.
   Parallel over blocks (OpenMP) — adjust pragmas to your build.
*/
static void apply_block_jacobi(const CSRMatrix *blocks, const uint32_t *ranges, uint32_t nb,
                                    const double *rhs_global, double *x_global,
                                    int maxit_block, double tol_block,
                                    double **block_prev_x /* optional, can be NULL */)
{
    /* 1) compute max block size for allocation */
    uint32_t max_s = 0;
    for(uint32_t b=0;b<nb;++b){
        uint32_t s = ranges[b+1] - ranges[b];
        if(s > max_s) max_s = s;
    }

    int nthreads = omp_get_max_threads();
    /* allocate per-thread work buffers once (contiguous arrays) */
    double **buf_b = (double**)malloc(sizeof(double*) * nthreads);
    double **buf_x = (double**)malloc(sizeof(double*) * nthreads);
    if(!buf_b || !buf_x){ fprintf(stderr,"OOM per-thread buffers\n"); exit(1); }
    for(int t=0;t<nthreads;++t){
        buf_b[t] = (double*)malloc(sizeof(double) * (size_t)max_s);
        buf_x[t] = (double*)malloc(sizeof(double) * (size_t)max_s);
        if(!buf_b[t] || !buf_x[t]){ fprintf(stderr,"OOM per-thread subbuffers\n"); exit(1); }
    }

    /* Parallel loop: each thread uses its buffers; avoid malloc/free inside loop */
    #pragma omp parallel for schedule(dynamic)
    for(int bi=0; bi<(int)nb; ++bi){
        int tid = omp_get_thread_num();
        double *b_local = buf_b[tid];
        double *x_local = buf_x[tid];

        uint32_t r0 = ranges[bi];
        uint32_t r1 = ranges[bi+1];
        uint32_t s = r1 - r0;

        /* fill b_local */
        for(uint32_t i=0;i<s;++i) b_local[i] = rhs_global[r0 + i];

        /* initial guess: if block_prev_x provided, use it as x_local initial guess (warm start),
           else zero initial guess (bicgstab_block currently zeros x_out anyway) */
        if(block_prev_x && block_prev_x[bi]){
            for(uint32_t i=0;i<s;++i) x_local[i] = block_prev_x[bi][i];
        } else {
            for(uint32_t i=0;i<s;++i) x_local[i] = 0.0;
        }

        /* call bicgstab_block (it will overwrite x_local). We deliberately use smaller inner iterations by default. */
        int ok = bicgstab_block(&blocks[bi], b_local, x_local, maxit_block, tol_block);

        if(ok != 0){
            /* fallback: diagonal scaling inside the block (cheap) */
            for(uint32_t i=0;i<s;++i){
                double diag = 0.0;
                uint32_t start = blocks[bi].indptr[i];
                uint32_t end   = blocks[bi].indptr[i+1];
                for(uint32_t p=start;p<end;++p) if(blocks[bi].indices[p] == i){ diag = blocks[bi].data[p]; break; }
                if(fabs(diag) < 1e-300) diag = 1.0;
                x_local[i] = b_local[i] / diag;
            }
        }

        /* copy result into global solution */
        for(uint32_t i=0;i<s;++i) x_global[r0 + i] = x_local[i];

        /* optionally save for warm start next apply */
        if(block_prev_x){
            if(block_prev_x[bi] == NULL){
                /* allocate storage for previous x if not present (single-threaded allocation is fine outside parallel region,
                   but we do it here conservatively using atomic to avoid race). Simpler path: assume block_prev_x already allocated by caller. */
            } else {
                for(uint32_t i=0;i<s;++i) block_prev_x[bi][i] = x_local[i];
            }
        }
    } /* end parallel */

    /* free per-thread buffers */
    for(int t=0;t<nthreads;++t){
        free(buf_b[t]);
        free(buf_x[t]);
    }
    free(buf_b); free(buf_x);
}

static bool save_csr_binary(const char* path, const CSRMatrix* M){
    FILE *f = fopen(path, "wb");
    if(!f){ perror("fopen save"); return false; }

    /* write header */
    if(fwrite(&M->rows, sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }
    if(fwrite(&M->nnz,  sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }

    /* write arrays */
    if(fwrite(M->indptr, sizeof(uint32_t), (size_t)(M->rows + 1), f) != (size_t)(M->rows + 1)){ fclose(f); return false; }
    if(M->nnz){
        if(fwrite(M->indices, sizeof(uint32_t), (size_t)M->nnz, f) != (size_t)M->nnz){ fclose(f); return false; }
        if(fwrite(M->data,    sizeof(double),   (size_t)M->nnz, f) != (size_t)M->nnz){ fclose(f); return false; }
    }
    fclose(f);
    return true;
}

static bool load_csr_binary(const char* path, CSRMatrix* M){
    FILE *f = fopen(path, "rb");
    if(!f) return false;

    uint32_t rows=0, nnz=0;
    if(fread(&rows, sizeof(uint32_t), 1, f) != 1){ fclose(f); return false; }
    if(fread(&nnz,  sizeof(uint32_t), 1, f) != 1){ fclose(f); return false; }

    /* allocate target CSR arrays using aligned allocator */
    M->rows = rows;
    M->nnz  = nnz;
    M->indptr = (uint32_t*) xaligned_malloc(sizeof(uint32_t)*(size_t)(rows+1));
    if(fread(M->indptr, sizeof(uint32_t), (size_t)(rows+1), f) != (size_t)(rows+1)){
        csr_free(M); fclose(f); return false;
    }

    if(nnz){
        M->indices = (uint32_t*) xaligned_malloc(sizeof(uint32_t)*(size_t)nnz);
        M->data    = (double*)    xaligned_malloc(sizeof(double)*(size_t)nnz);
        if(fread(M->indices, sizeof(uint32_t), (size_t)nnz, f) != (size_t)nnz){
            csr_free(M); fclose(f); return false;
        }
        if(fread(M->data, sizeof(double), (size_t)nnz, f) != (size_t)nnz){
            csr_free(M); fclose(f); return false;
        }
    } else {
        /* safe defaults for empty matrix */
        M->indices = NULL;
        M->data    = NULL;
    }

    fclose(f);
    return true;
}



// ===================== SpMV =====================

// y += A*x, iterating by row stripes to improve cache locality
static void band_apply_add(const CSRMatrix* A, const double* __restrict x, double* __restrict y){
    const uint32_t n = A->rows;
    // Use cache hints to set stripe size if not explicitly set
    const uint32_t stripe = STRIPE_ROWS ? STRIPE_ROWS : (BIGCACHE_SZ / (sizeof(double)*8));
    #pragma omp parallel for schedule(static)
    for(int64_t s = 0; s < (int64_t)n; s += stripe){
        uint32_t iend = (uint32_t)((s + stripe) > n ? n : (s + stripe));
        for(uint32_t i=(uint32_t)s; i<iend; ++i){
            double sum=0.0;
            uint32_t row_start = A->indptr[i];
            uint32_t row_end = A->indptr[i+1];
            for(uint32_t k=row_start; k<row_end; ++k){ sum += A->data[k]*x[A->indices[k]]; }
            y[i] += sum;
        }
    }
}


void compute_print_true_residual(const CSRMatrix* Band, const CSRMatrix* Other,
                                 const Vector* b, const Vector* x){
    uint32_t n = b->size;
    Vector Ax = vec_empty(n);
    vzero(Ax.ptr, n);
    if(Band && Band->nnz) band_apply_add(Band, x->ptr, Ax.ptr); // Ax += Band * x
    if(Other && Other->nnz){ Vector tmp = vec_empty(n); vzero(tmp.ptr,n); csr_apply(Other, x->ptr, tmp.ptr);
        #pragma omp parallel for
        for(uint32_t i=0;i<n;++i) Ax.ptr[i]+=tmp.ptr[i];
    }
    // r_true = b - Ax  (reuse Ax to store residual)
    #pragma omp parallel for
    for(uint32_t i=0;i<n;++i) Ax.ptr[i] = b->ptr[i] - Ax.ptr[i];
    double tr = vnorm2(Ax.ptr, n);
    double bn = fmax(1e-30, vnorm2(b->ptr,n));
    printf("TRUE relative residual: %.6e\n", tr/bn);
    vec_free(&Ax);
}

void check_AM_is_I(const CSRMatrix *Band, const CSRMatrix *Other, const CSRMatrix *Minv, uint32_t N){
    Vector e = vec_empty(N), tmp = vec_empty(N), out = vec_empty(N);
    srand(123);
    int NS = 10;
    double maxerr = 0.0;
    for(int s=0;s<NS;++s){
        int j = rand() % N;
        vzero(e.ptr, N); e.ptr[j]=1.0;
        /* tmp = Minv * e */
        csr_apply(Minv, e.ptr, tmp.ptr);
        /* out = A * tmp */
        vzero(out.ptr, N);
        if(Band && Band->nnz) band_apply_add(Band, tmp.ptr, out.ptr);
        if(Other && Other->nnz){ Vector tmp2 = vec_empty(N); vzero(tmp2.ptr,N); csr_apply(Other, tmp.ptr, tmp2.ptr);
            #pragma omp parallel for
            for(uint32_t i=0;i<N;++i) out.ptr[i]+=tmp2.ptr[i];
            vec_free(&tmp2);
        }
        out.ptr[j] -= 1.0; /* A*(M*e_j) - e_j */
        double err = vnorm2(out.ptr, N);
        printf("AM-I test j=%d err = %.6e\n", j, err);
        maxerr = fmax(maxerr, err);
    }
    printf("AM-I max(sample) = %.6e\n", maxerr);
    vec_free(&e); vec_free(&tmp); vec_free(&out);
}
// ===================== BiCGStab (left preconditioning with SPAI inverse) =====================
static int bicgstab_solve(const CSRMatrix* Band, const CSRMatrix* Other, const CSRMatrix* Minv,
                          const Vector* b, Vector* x, double tol, unsigned int maxit,
                          unsigned int* iters, double* out_final_res){
    
    compute_print_true_residual(Band, Other, b, x);
    const uint32_t n = b->size;
    Vector r = vec_empty(n), r0 = vec_empty(n), p = vec_empty(n), v = vec_empty(n);
    Vector t = vec_empty(n), s = vec_empty(n), z = vec_empty(n), y = vec_empty(n);
    Vector tmp = vec_empty(n);

    // r0 = b - A*x
    vzero(r.ptr,n);
    
    if(Band && Band->nnz) band_apply_add(Band, x->ptr, r.ptr);
    

    if(Other && Other->nnz){
        vzero(tmp.ptr,n); 
        csr_apply(Other, x->ptr, tmp.ptr);
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) r.ptr[i]+=tmp.ptr[i];
    }
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) r.ptr[i] = b->ptr[i] - r.ptr[i];
    memcpy(r0.ptr, r.ptr, sizeof(double)*n); // <-- исправление


    double rho=1.0, alpha=1.0, omega=1.0;
    vzero(p.ptr,n);
    vzero(v.ptr,n);

    double bnrm2 = fmax(1e-30, vnorm2(b->ptr,n));
    double resid = vnorm2(r.ptr,n)/bnrm2;
    if(resid < tol){ *iters = 0; *out_final_res = resid; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); return 0; }

    for(unsigned int it=1; it<=maxit; ++it){
        
        double rho1 = vdot(r0.ptr, r.ptr, n);
        if (it % 10 == 0){
           printf("Iteration %u: relative residual %.6e\n", it, resid);
            compute_print_true_residual(Band, Other, b, x);
            printf("it=%u rho1=%.3e alpha=%.3e omega=%.3e resid=%.3e\n", it, rho1, alpha, omega, resid);
            double max_abs = 0.0;
            for(uint32_t i=0;i<Minv->nnz;++i) max_abs = fmax(max_abs, fabs(Minv->data[i]));
            printf("Minv nnz=%u, max|Minv|=%.3e\n", Minv->nnz, max_abs);
            double maxv=0.0; size_t nz = Minv->nnz;
            double sumabs=0.0;
            for(size_t k=0;k<nz;++k){ double a=fabs(Minv->data[k]); maxv=fmax(maxv,a); sumabs += a; }
            printf("Minv nnz=%u max|Minv|=%.3e mean|val|=%.3e\n", Minv->nnz, maxv, sumabs / (double)nz);
        }
        if(fabs(rho1) < 1e-300) { *iters = it-1; break; }
        double beta = (rho1/rho)*(alpha/omega);
        // p = r + beta*(p - omega*v)
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) p.ptr[i] = r.ptr[i] + beta*(p.ptr[i] - omega*v.ptr[i]);
#if 0
        apply_precond(Minv, p.ptr, y.ptr);
#else
        apply_block_jacobi(GBlockPrecond.blocks, GBlockPrecond.ranges, GBlockPrecond.nb,
        p.ptr, y.ptr, 30, 1e-2, NULL);
#endif

        vzero(v.ptr,n);
        if(Band && Band->nnz) band_apply_add(Band, y.ptr, v.ptr);
        if(Other && Other->nnz){ 
            vzero(tmp.ptr,n);
            csr_apply(Other, y.ptr, tmp.ptr); 
            #pragma omp parallel for schedule(static)
            for(uint32_t i=0;i<n;++i) v.ptr[i]+=tmp.ptr[i]; 
        }

        double r0v = vdot(r0.ptr, v.ptr, n);
        if(fabs(r0v) < 1e-300){ *iters = it-1; break; }
        alpha = rho1 / r0v;

        // s = r - alpha*v
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) s.ptr[i] = r.ptr[i] - alpha*v.ptr[i];

        // Early convergence check
        double snrm = vnorm2(s.ptr,n)/bnrm2;
        if(snrm < tol){ // x = x + alpha*y
            vaxpy(x->ptr, alpha, y.ptr, n);
            *iters = it; *out_final_res = snrm; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); vec_free(&tmp); return 0; }

        // z = M * s
#if 0
        apply_precond(Minv, s.ptr, z.ptr);
#else
        apply_block_jacobi(GBlockPrecond.blocks, GBlockPrecond.ranges, GBlockPrecond.nb,
        s.ptr, z.ptr,  30, 1e-2, NULL);
#endif

        // t = A*z
        vzero(t.ptr,n);
        if(Band && Band->nnz) band_apply_add(Band, z.ptr, t.ptr);
        if(Other && Other->nnz){ 
            vzero(tmp.ptr,n); 
            csr_apply(Other, z.ptr, tmp.ptr); 
            #pragma omp parallel for schedule(static)
            for(uint32_t i=0;i<n;++i) t.ptr[i]+=tmp.ptr[i];
        }

        double tt = vdot(t.ptr,t.ptr,n);
        if(tt < 1e-300){ *iters = it-1; break; }
        double ts = vdot(t.ptr,s.ptr,n);
        omega = ts/tt;

        // x = x + alpha*y + omega*z
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) x->ptr[i] += alpha*y.ptr[i] + omega*z.ptr[i];

        // r = s - omega*t
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) r.ptr[i] = s.ptr[i] - omega*t.ptr[i];

        resid = vnorm2(r.ptr,n)/bnrm2;
        if(resid < tol){ *iters = it; *out_final_res = resid; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); vec_free(&tmp); return 0; }

        if(fabs(omega) < 1e-300){ *iters = it; break; }
        rho = rho1;
    }

    *out_final_res = resid; 
    vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); vec_free(&tmp);
    return 1; // not fully converged
}

// ===================== Main =====================
int main(int argc, char** argv){
    const char* MATFILE = "pwtk.mat";
    const char* STRUCT  = "Problem"; 
    const char* FIELD   = "A";      
    const char* BFILE   = "b.bin";
    const char* XFILE   = "x.bin";
    const char* MINV_FILE = "minv3.bin";
    if(argc==4 || argc==9){
        MATFILE = argv[1]; STRUCT = argv[2]; FIELD = argv[3];
    }
    if(argc==9){
        TOLERANCE = strtod(argv[4], NULL);
        MAXITER   = (unsigned)strtoul(argv[5], NULL, 10);
        THREADS_NUM = (unsigned)strtoul(argv[6], NULL, 10);
        STRIPE_ROWS = (unsigned)strtoul(argv[7], NULL, 10);
        BW = (unsigned)strtoul(argv[8], NULL, 10);
    }
    calculate_stripe_rows(BW);
    omp_set_num_threads((int)THREADS_NUM);

    mat_t *file = Mat_Open(MATFILE, MAT_ACC_RDONLY);
    if(!file){ fprintf(stderr, "No file has been found\\n"); puts("1"); return 1; }
    matvar_t *var = Mat_VarRead(file, STRUCT); Mat_Close(file);
    if(!var){ fprintf(stderr, "No variable has been found\\n"); puts("2"); return 2; }
    if(var->class_type != MAT_C_STRUCT){ fprintf(stderr, "No struct has been found\\n"); Mat_VarFree(var); puts("3"); return 3; }
    matvar_t *field = Mat_VarGetStructFieldByName(var, FIELD, 0);
    if(!field){ fprintf(stderr, "No field has been found\\n"); Mat_VarFree(var); puts("4"); return 4; }
    if(field->class_type != MAT_C_SPARSE || field->data_type != MAT_T_DOUBLE || field->data == NULL){
        fprintf(stderr, "Matrix is wrong\\n"); Mat_VarFree(field); Mat_VarFree(var); puts("5"); return 5;
    }

    mat_sparse_t *input = (mat_sparse_t*)field->data;
    const uint32_t N = (uint32_t)field->dims[0];

    CSRMatrix A_band={}, A_other={};
    csc_to_csr_and_split(input, N, BW, &A_band, &A_other);

    CSRMatrix Minv = {0};

#if 1
    if(load_csr_binary(MINV_FILE, &Minv)){
        printf("Loaded preconditioner from %s\n", MINV_FILE);
    } else {
        printf("Preconditioner file not found — building...\n");
        //spai0_build_diagonal(input, N, &Minv);
        //spai_build_improved(input, N, &Minv); 
        //spai0_build_banded(input, N, &Minv);
        spai_build_blockdiag(input, N, &Minv);
        uint32_t Bsize = 6400; /* или 6400/подберите */ 
        CSRMatrix *blocks = NULL;
        uint32_t *block_ranges = NULL;
        uint32_t nb_blocks = 0;
        build_block_jacobi_blocks(input, N, Bsize, &blocks, &block_ranges, &nb_blocks);

        /* store in global for solver to use */
        GBlockPrecond.blocks = blocks;
        GBlockPrecond.ranges = block_ranges;
        GBlockPrecond.nb = nb_blocks;

        /* optional: print basic stats (total nnz in blocks) */
        uint64_t total_nnz = 0;
        for(uint32_t b=0;b<nb_blocks;++b) total_nnz += (uint64_t)blocks[b].indptr[blocks[b].rows];
        printf("Block-Jacobi built: nb=%u, total block nnz=%" PRIu64 "\n", nb_blocks, total_nnz);
        printf("Preconditioner built. nnz=%u\n", Minv.nnz);
        if(!save_csr_binary(MINV_FILE, &Minv)){
            fprintf(stderr, "Warning: failed to save preconditioner to %s\n", MINV_FILE);
        } else {
            printf("Preconditioner saved to %s\n", MINV_FILE);
        }
        check_AM_is_I(&A_band, &A_other, &Minv, N);
    }

#else
    Minv = csr_empty(N, N);
    for(uint32_t i=0;i<N;++i){ Minv.indptr[i]=i; Minv.indices[i]=i; Minv.data[i]=1.0; }
    Minv.indptr[N] = N;
    printf("DEBUG: using identity preconditioner (should reduce to unpreconditioned BiCGStab)\n");
#endif

    Vector b = make_or_load_b(BFILE, N);
    

    Vector x = vec_empty(N); vzero(x.ptr, N);
    

    double t0 = wall_seconds();
    unsigned int iters=0; double final_res=0.0;
    

    int status = bicgstab_solve(&A_band, &A_other, &Minv, &b, &x, TOLERANCE, MAXITER, &iters, &final_res);

    double t1 = wall_seconds();

    printf("Converged: %s\n", status==0?"yes":"no");
    printf("Iterations: %u\n", iters);
    printf("Final relative residual: %.6e\n", final_res);
    printf("Elapsed (s): %.6f\n", (t1-t0));

    // оценка

    Vector tmp1 = vec_empty(N);
    Vector tmp2 = vec_empty(N);
    vzero(tmp1.ptr, N);
    if(A_band.nnz) band_apply_add(&A_band, x.ptr, tmp1.ptr);
    if(A_other.nnz){ 
        vzero(tmp2.ptr, N); 
        csr_apply(&A_other, x.ptr, tmp2.ptr); 
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0; i<N; ++i) tmp2.ptr[i]+=tmp1.ptr[i];
    }
    vdif(tmp2.ptr, b.ptr, tmp1.ptr, N);
    printf("Residual norm: %lf\n", vnorm2(tmp1.ptr, N));

    vec_free(&tmp1);
    vec_free(&tmp2);
    write_binary_vector(XFILE, &x);

    vec_free(&x); vec_free(&b);
    csr_free(&A_band); csr_free(&A_other); 
    
    csr_free(&Minv);
#if 1
    if(GBlockPrecond.blocks){
    for(uint32_t b=0;b<GBlockPrecond.nb;++b){
        free(GBlockPrecond.blocks[b].indptr);
        free(GBlockPrecond.blocks[b].indices);
        free(GBlockPrecond.blocks[b].data);
    }
    free(GBlockPrecond.blocks);
    free(GBlockPrecond.ranges);
    GBlockPrecond.blocks = NULL;
    GBlockPrecond.ranges = NULL;
    GBlockPrecond.nb = 0;
}
#endif
    Mat_VarFree(field); Mat_VarFree(var);
    return status;
}
