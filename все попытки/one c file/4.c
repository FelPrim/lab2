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

static double TOLERANCE = 1e-9;
static unsigned int MAXITER = 1000;
static unsigned int THREADS_NUM = 12;
static unsigned int BIGCACHE_SZ = 512000;
static unsigned int BW = 16;              
unsigned int STRIPE_ROWS = 0;

static inline void calculate_stripe_rows(size_t NNZ, size_t ROWS){
    size_t nz_row = NNZ / ROWS;          // среднее ненулевых на строку (float)
    size_t bytes_per_nonzero = sizeof(double) + sizeof(uint32_t);  // ~8 + 4 = 12 bytes
    size_t bytes_per_row = nz_row * bytes_per_nonzero + sizeof(double); // + место для y[i]
    double usable_fraction = 0.7;      // доля L2, которую мы реально хотим занять (0.4-0.7)

    STRIPE_ROWS = (unsigned int) ((BIGCACHE_SZ * usable_fraction) / bytes_per_row);

    //STRIPE_ROWS = BIGCACHE_SZ/(16*bw) - 1;
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
static void csr_apply(const CSRMatrix* A, const double* restrict x, double* restrict y){
    const uint32_t n = A->rows; 
    #pragma omp parallel for schedule(guided)
    for(uint32_t i=0;i<n;++i){
        double sum=0.0;
        for(uint32_t k=A->indptr[i]; k<A->indptr[i+1]; ++k){ sum += A->data[k]*x[A->indices[k]]; }
        y[i]=sum;
    }
}

// y += A*x, iterating by row stripes to improve cache locality
static void band_apply_add(const CSRMatrix* A, const double* restrict x, double* restrict y){
    const uint32_t n = A->rows;
    // Use cache hints to set stripe size if not explicitly set
    const uint32_t stripe = STRIPE_ROWS ? STRIPE_ROWS : (BIGCACHE_SZ / (sizeof(double)*8));
    #pragma omp parallel for schedule(guided)
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


// Apply approximate inverse M (stored as CSR) : z = M * r
static void apply_precond(const CSRMatrix* M, const double* restrict r, double* restrict z){
    csr_apply(M, r, z);
}


void compute_print_true_residual(const CSRMatrix* Band, const CSRMatrix* Other,
                                 const Vector* b, const Vector* x){
    uint32_t n = b->size;
    Vector Ax = vec_empty(n);
    vzero(Ax.ptr, n);
    if(Band && Band->nnz) band_apply_add(Band, x->ptr, Ax.ptr); // Ax += Band * x
    if(Other && Other->nnz){ Vector tmp = vec_empty(n); vzero(tmp.ptr,n); csr_apply(Other, x->ptr, tmp.ptr);
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) Ax.ptr[i]+=tmp.ptr[i];
    }
    // r_true = b - Ax  (reuse Ax to store residual)
    #pragma omp parallel for schedule(static)
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
            #pragma omp parallel for schedule(static)
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
    
    //compute_print_true_residual(Band, Other, b, x);
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
   // if(resid < tol){ *iters = 0; *out_final_res = resid; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); return 0; }

    for(unsigned int it=1; it<=maxit; ++it){
        
        double rho1 = vdot(r0.ptr, r.ptr, n);
       // if (it % 100 == 0){
        //    printf("it=%u rho1=%.3e alpha=%.3e omega=%.3e resid=%.3e\n", it, rho1, alpha, omega, resid);
         //   double max_abs = 0.0;
        //}
     
        //if(fabs(rho1) < 1e-300) { *iters = it-1; break; }
        double beta = (rho1/rho)*(alpha/omega);
        // p = r + beta*(p - omega*v)
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) p.ptr[i] = r.ptr[i] + beta*(p.ptr[i] - omega*v.ptr[i]);

        apply_precond(Minv, p.ptr, y.ptr);

        vzero(v.ptr,n);
        if(Band && Band->nnz) band_apply_add(Band, y.ptr, v.ptr);
        if(Other && Other->nnz){ 
            vzero(tmp.ptr,n);
            csr_apply(Other, y.ptr, tmp.ptr); 
            #pragma omp parallel for schedule(static)
            for(uint32_t i=0;i<n;++i) v.ptr[i]+=tmp.ptr[i]; 
        }

        double r0v = vdot(r0.ptr, v.ptr, n);
        //if(fabs(r0v) < 1e-300){ *iters = it-1; break; }
        alpha = rho1 / r0v;

        // s = r - alpha*v
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) s.ptr[i] = r.ptr[i] - alpha*v.ptr[i];

        // Early convergence check
        double snrm = vnorm2(s.ptr,n)/bnrm2;
        // if(snrm < tol){ // x = x + alpha*y
        //    vaxpy(x->ptr, alpha, y.ptr, n);
        //    *iters = it; *out_final_res = snrm; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); vec_free(&tmp); return 0; }

        // z = M * s
        apply_precond(Minv, s.ptr, z.ptr);

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
        // if(tt < 1e-300){ *iters = it-1; break; }
        double ts = vdot(t.ptr,s.ptr,n);
        omega = ts/tt;

        // x = x + alpha*y + omega*z
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) x->ptr[i] += alpha*y.ptr[i] + omega*z.ptr[i];

        // r = s - omega*t
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i) r.ptr[i] = s.ptr[i] - omega*t.ptr[i];

        resid = vnorm2(r.ptr,n)/bnrm2;
        // if(resid < tol){ *iters = it; *out_final_res = resid; vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y); vec_free(&tmp); return 0; }

        // if(fabs(omega) < 1e-300){ *iters = it; break; }
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
    const char* MINV_FILE = "minv4.bin";
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
    
    calculate_stripe_rows(input->nzmax, N);
    printf("stripe rows: %u\n", STRIPE_ROWS);
    csc_to_csr_and_split(input, N, BW, &A_band, &A_other);

    CSRMatrix Minv = {0};

#if 1
    if(load_csr_binary(MINV_FILE, &Minv)){
       // printf("Loaded preconditioner from %s\n", MINV_FILE);
    } else {
        printf("Preconditioner file not found — building...\n");
        spai0_build_diagonal(input, N, &Minv);
        //spai_build_improved(input, N, &Minv); 
        //spai0_build_banded(input, N, &Minv);

        //spai_build_blockdiag(input, N, &Minv);

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

    // printf("Converged: %s\n", status==0?"yes":"no");
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
    csr_free(&A_band); csr_free(&A_other); csr_free(&Minv);
    Mat_VarFree(field); Mat_VarFree(var);
    return status;
}
