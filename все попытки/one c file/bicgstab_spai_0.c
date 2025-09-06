#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <matio.h>

#if defined(_WIN32)
#include <malloc.h>
static inline void *aligned_alloc( size_t alignment, size_t size){
    return _aligned_malloc(size, alignment);
}
#endif

#ifndef ALIGN
#define ALIGN 64
#endif

#ifndef likely
#define likely(x)   __builtin_expect(!!(x),1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x),0)
#endif

// ===================== Data structures =====================
typedef struct Vector{
    alignas(32) double *ptr;
    uint32_t size;
} Vector;

typedef struct CSRMatrix{
    alignas(32) double *data;
    alignas(32) uint32_t *indices;
    uint32_t *indptr; // length rows+1
    uint32_t rows;     // == cols for our use
    uint32_t nnz;
} CSRMatrix;

// ===================== Parameters (overridable via argv) =====================
static double TOLERANCE = 1e-9;
static unsigned int MAXITER = 10000;
static unsigned int THREADS_NUM = 2;
static unsigned int STRIPE_ROWS = 256;     // rows per cache stripe inside band_apply
static unsigned int BW = 64;               // half-bandwidth for splitting A (|i-j| <= BW -> band)

// ===================== Utility =====================
static inline double wall_seconds(){ return omp_get_wtime(); }

static void* xaligned_malloc(size_t nbytes){
    void *p = NULL;
#if defined(_WIN32)
    p = _aligned_malloc(nbytes, ALIGN);
#else
    if(posix_memalign(&p, ALIGN, nbytes)) p = NULL;
#endif
    if (!p){ fprintf(stderr, "OOM for %zu bytes\n", nbytes); exit(1);} 
    return p;
}

static void csr_free(CSRMatrix *A){
    if(!A) return;
    free(A->data); free(A->indices); free(A->indptr);
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
    Vector v; v.size=n; v.ptr=(double*)xaligned_malloc(sizeof(double)*n);
    return v;
}

static void vec_free(Vector *v){ if(!v) return; free(v->ptr); v->ptr=NULL; v->size=0; }

// ===================== Basic BLAS-like ops =====================
static inline void vcopy(const double* __restrict a, double* __restrict b, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) b[i]=a[i];
}
static inline void vzero(double* __restrict a, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) a[i]=0.0;
}
static inline void vaxpy(double* __restrict y, double a, const double* __restrict x, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) y[i] += a*x[i];
}
static inline void vxpay(double* __restrict y, double b, const double* __restrict x, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) y[i] = x[i] + b*y[i];
}
static inline void vscale(double* __restrict y, double a, uint32_t n){
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) y[i] *= a;
}
static inline double vdot(const double* __restrict a, const double* __restrict b, uint32_t n){
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for(uint32_t i=0;i<n;++i) sum += a[i]*b[i];
    return sum;
}
static inline double vnorm2(const double* __restrict a, uint32_t n){ return sqrt(vdot(a,a,n)); }

// ===================== I/O for b and x =====================
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
    unsigned long seed = 42ul;
    #pragma omp parallel
    {
        unsigned int s = (unsigned int)(seed ^ (unsigned long)omp_get_thread_num());
        #pragma omp for schedule(static)
        for(uint32_t i=0;i<n;++i){ b.ptr[i] = rand_r(&s)/((double)RAND_MAX) - 0.5; }
    }
    write_binary_vector(path,&b);
    return b;
}

// ===================== matio (CSC) -> CSR + split into band/off-band =====================
static void csc_to_csr_and_split(const mat_sparse_t* A, uint32_t N, unsigned int bw,
                                 CSRMatrix* Band, CSRMatrix* Other){
    const uint32_t* jc = (const uint32_t*)A->jc; // size N+1
    const uint32_t* ir = (const uint32_t*)A->ir; // row indices of each csc nnz
    const double*  pr = (const double*) A->data;
    const uint32_t nnz = (uint32_t)A->ndata;

    // Count band/off-band per row
    uint32_t *row_band = (uint32_t*)calloc(N,sizeof(uint32_t));
    uint32_t *row_other= (uint32_t*)calloc(N,sizeof(uint32_t));
    for(uint32_t col=0; col<N; ++col){
        for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
            uint32_t r = ir[k];
            if((int)abs((int)r-(int)col) <= (int)bw) row_band[r]++; else row_other[r]++;
        }
    }
    // Prefix sums -> indptr
    *Band  = csr_empty(N, 0); 
    *Other = csr_empty(N, 0);
    Band->indptr[0]=0; Other->indptr[0]=0;
    for(uint32_t i=0;i<N;++i){
        Band->indptr[i+1]  = Band->indptr[i]  + row_band[i];
        Other->indptr[i+1] = Other->indptr[i] + row_other[i];
    }
    Band->nnz  = Band->indptr[N];
    Other->nnz = Other->indptr[N];
    free(row_band); free(row_other);
    // allocate data
    free(Band->data); free(Band->indices); // from csr_empty(0)
    free(Other->data); free(Other->indices);
    Band->data    = (double*)xaligned_malloc(sizeof(double)*Band->nnz);
    Band->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*Band->nnz);
    Other->data    = (double*)xaligned_malloc(sizeof(double)*Other->nnz);
    Other->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*Other->nnz);

    // row-wise cursors
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
    free(wb); free(wo);
}

// ===================== SPAI(0) builder (diagonal-only variant) =====================
// We build an approximate inverse with diagonal sparsity: M = diag(1/diag(A)).
// This is a minimal SPAI(0) pattern; extend with neighbors to enrich if needed.
static void spai0_build_diagonal(const mat_sparse_t* A, uint32_t N, CSRMatrix* M){
    const uint32_t* jc = (const uint32_t*)A->jc; // size N+1
    const uint32_t* ir = (const uint32_t*)A->ir; 
    const double*  pr = (const double*) A->data;

    *M = csr_empty(N, N); // one nnz per row
    for(uint32_t i=0;i<=N;++i) M->indptr[i]=i; // exactly one per row

    for(uint32_t i=0;i<N;++i){
        double aii = 0.0;
        for(uint32_t k=jc[i]; k<jc[i+1]; ++k){ if(ir[k]==i){ aii = pr[k]; break; } }
        M->indices[i] = i;
        M->data[i] = (fabs(aii)>0)? 1.0/aii : 1.0; // fallback if zero-diagonal
    }
}

// ===================== SpMV =====================
static void csr_apply(const CSRMatrix* A, const double* __restrict x, double* __restrict y){
    const uint32_t n = A->rows; 
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i){
        double sum=0.0;
        for(uint32_t k=A->indptr[i]; k<A->indptr[i+1]; ++k){ sum += A->data[k]*x[A->indices[k]]; }
        y[i]=sum;
    }
}

// y += A*x, iterating by row stripes to improve cache locality
static void band_apply_add(const CSRMatrix* A, const double* __restrict x, double* __restrict y){
    const uint32_t n = A->rows; 
    const uint32_t stripe = STRIPE_ROWS ? STRIPE_ROWS : 128;
    #pragma omp parallel for schedule(static)
    for(int64_t s = 0; s < (int64_t)n; s += stripe){
        uint32_t iend = (uint32_t)((s + stripe) > n ? n : (s + stripe));
        for(uint32_t i=(uint32_t)s; i<iend; ++i){
            double sum=0.0;
            uint32_t row_start = A->indptr[i];
            uint32_t row_end   = A->indptr[i+1];
            // Inner loop is tight and cache-friendly as x access clusters near diagonal when A is banded
            for(uint32_t k=row_start; k<row_end; ++k){ sum += A->data[k]*x[A->indices[k]]; }
            y[i] += sum;
        }
    }
}

// Apply approximate inverse M (stored as CSR) : z = M * r
static void apply_precond(const CSRMatrix* M, const double* __restrict r, double* __restrict z){
    csr_apply(M, r, z);
}

// ===================== BiCGStab (left preconditioning with SPAI inverse) =====================
static int bicgstab_solve(const CSRMatrix* Band, const CSRMatrix* Other, const CSRMatrix* Minv,
                          const Vector* b, Vector* x, double tol, unsigned int maxit,
                          unsigned int* iters, double* out_final_res){
    const uint32_t n = b->size;
    Vector r = vec_empty(n), r0 = vec_empty(n), p = vec_empty(n), v = vec_empty(n);
    Vector t = vec_empty(n), s = vec_empty(n), z = vec_empty(n), y = vec_empty(n);

    // r0 = b - A*x
    vzero(r.ptr,n);
    if(Band && Band->nnz) band_apply_add(Band, x->ptr, r.ptr);
    if(Other && Other->nnz){
        Vector tmp = vec_empty(n); vzero(tmp.ptr,n); csr_apply(Other, x->ptr, tmp.ptr);
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) r.ptr[i]+=tmp.ptr[i];
        vec_free(&tmp);
    }
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) r.ptr[i] = b->ptr[i] - r.ptr[i];
    vcopy(r.ptr, r0.ptr, n);

    double rho=1.0, alpha=1.0, omega=1.0;
    vzero(p.ptr,n);
    vzero(v.ptr,n);

    double bnrm2 = fmax(1e-30, vnorm2(b->ptr,n));
    double resid = vnorm2(r.ptr,n)/bnrm2;
    if(resid < tol){ *iters = 0; *out_final_res = resid; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); return 0; }

    for(unsigned int it=1; it<=maxit; ++it){
        double rho1 = vdot(r0.ptr, r.ptr, n);
        if(fabs(rho1) < 1e-300) { *iters = it-1; break; }
        double beta = (rho1/rho)*(alpha/omega);
        // p = r + beta*(p - omega*v)
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) p.ptr[i] = r.ptr[i] + beta*(p.ptr[i] - omega*v.ptr[i]);

        // preconditioned direction y = M * p  (SPAI is approx inverse)
        apply_precond(Minv, p.ptr, y.ptr);

        // v = A*y
        vzero(v.ptr,n);
        if(Band && Band->nnz) band_apply_add(Band, y.ptr, v.ptr);
        if(Other && Other->nnz){ Vector tmp = vec_empty(n); vzero(tmp.ptr,n); csr_apply(Other, y.ptr, tmp.ptr); 
            #pragma omp parallel for schedule(static)
            for(uint32_t i=0;i<n;++i) v.ptr[i]+=tmp.ptr[i]; vec_free(&tmp); }

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
            *iters = it; *out_final_res = snrm; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); return 0; }

        // z = M * s
        apply_precond(Minv, s.ptr, z.ptr);

        // t = A*z
        vzero(t.ptr,n);
        if(Band && Band->nnz) band_apply_add(Band, z.ptr, t.ptr);
        if(Other && Other->nnz){ Vector tmp2 = vec_empty(n); vzero(tmp2.ptr,n); csr_apply(Other, z.ptr, tmp2.ptr); 
            #pragma omp parallel for schedule(static)
            for(uint32_t i=0;i<n;++i) t.ptr[i]+=tmp2.ptr[i]; vec_free(&tmp2); }

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
        if(resid < tol){ *iters = it; *out_final_res = resid; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); return 0; }

        if(fabs(omega) < 1e-300){ *iters = it; break; }
        rho = rho1;
    }

    *out_final_res = resid; 
    vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y);
    return 1; // not fully converged
}

// ===================== Main =====================
int main(int argc, char** argv){
    // Defaults for MATLAB .mat input
    const char* MATFILE = "pwtk.mat";
    const char* STRUCT  = "Problem"; 
    const char* FIELD   = "A";      // sparse matrix in CSC inside struct
    const char* BFILE   = "b.bin";
    const char* XFILE   = "x.bin";

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
    omp_set_num_threads((int)THREADS_NUM);

    // ---- Load matrix from .mat ----
    mat_t *file = Mat_Open(MATFILE, MAT_ACC_RDONLY);
    if(!file){ fprintf(stderr, "No file has been found\n"); return 1; }
    matvar_t *var = Mat_VarRead(file, STRUCT); Mat_Close(file);
    if(!var){ fprintf(stderr, "No variable has been found\n"); return 2; }
    if(var->class_type != MAT_C_STRUCT){ fprintf(stderr, "No struct has been found\n"); Mat_VarFree(var); return 3; }
    matvar_t *field = Mat_VarGetStructFieldByName(var, FIELD, 0);
    if(!field){ fprintf(stderr, "No field has been found\n"); Mat_VarFree(var); return 4; }
    if(field->class_type != MAT_C_SPARSE || field->data_type != MAT_T_DOUBLE || field->data == NULL){
        fprintf(stderr, "Matrix is wrong\n"); Mat_VarFree(field); Mat_VarFree(var); return 5;
    }

    mat_sparse_t *input = (mat_sparse_t*)field->data;
    const uint32_t N = (uint32_t)field->dims[0];

    // ---- Split A into band/off-band in CSR ----
    CSRMatrix A_band={}, A_other={};
    csc_to_csr_and_split(input, N, BW, &A_band, &A_other);

    // ---- Build SPAI(0) preconditioner (diagonal variant) ----
    CSRMatrix Minv={};
    spai0_build_diagonal(input, N, &Minv);

    // ---- Build/load b ----
    Vector b = make_or_load_b(BFILE, N);

    // ---- Solve with BiCGStab ----
    Vector x = vec_empty(N); vzero(x.ptr, N);

    double t0 = wall_seconds();
    unsigned int iters=0; double final_res=0.0;
    int status = bicgstab_solve(&A_band, &A_other, &Minv, &b, &x, TOLERANCE, MAXITER, &iters, &final_res);
    double t1 = wall_seconds();

    // ---- Output ----
    printf("Converged: %s\n", status==0?"yes":"no");
    printf("Iterations: %u\n", iters);
    printf("Final relative residual: %.6e\n", final_res);
    printf("Elapsed (s): %.6f\n", (t1-t0));

    // Save solution
    write_binary_vector(XFILE, &x);

    // Cleanup
    vec_free(&x); vec_free(&b);
    csr_free(&A_band); csr_free(&A_other); csr_free(&Minv);
    Mat_VarFree(field); Mat_VarFree(var);
    return status;
}
