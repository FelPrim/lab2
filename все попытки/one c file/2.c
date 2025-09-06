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

//////////////////////////////////////////////
static double TOLERANCE = 1e-9;
static unsigned int MAXITER = 1000;
static unsigned int THREADS_NUM = 12;
static unsigned int BIGCACHE_SZ = 512000;
static unsigned int BW = 16;              
unsigned int STRIPE_ROWS = 0;
/////////////////////////////////////////////
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

static void spai0_build_banded(const mat_sparse_t* A, uint32_t N, CSRMatrix* M){
    

    const uint32_t* jc = (const uint32_t*)A->jc; // size N+1
    const uint32_t* ir = (const uint32_t*)A->ir;
    const double*  pr = (const double*) A->data;

    /* cap half-bandwidth at 128 */
    const uint32_t bw_spai = (BW < 128 ? BW : 128);

    /* extract diagonal (fallback to 1.0 when missing/zero) */
    double *diag = (double*)malloc(sizeof(double)*N);
    if(!diag){ fprintf(stderr, "OOM diag\n"); exit(1); }
    for(uint32_t i=0;i<N;++i) diag[i] = 0.0;
    for(uint32_t col=0; col<N; ++col){
        for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
            uint32_t r = ir[k];
            if(r==col){ diag[col] = pr[k]; break; }
        }
    }
    for(uint32_t i=0;i<N;++i) if(fabs(diag[i]) < 1e-300) diag[i] = 1.0;
    
    /* count nnz per row for M: only where A has entries inside the band */
    uint32_t *row_count = (uint32_t*)calloc(N, sizeof(uint32_t));
    if(!row_count){ fprintf(stderr, "OOM row_count\n"); exit(1); }

    for(uint32_t j=0;j<N;++j){
        uint32_t lo = (j > bw_spai) ? j - bw_spai : 0;
        uint32_t hi = (j + bw_spai < N-1) ? j + bw_spai : N-1;
        for(uint32_t k=jc[j]; k<jc[j+1]; ++k){
            uint32_t r = ir[k];
            if(r < lo || r > hi) continue;
            row_count[r]++;
        }
    }
    /* ensure at least diagonal entry per row */
    for(uint32_t i=0;i<N;++i) if(row_count[i] == 0) row_count[i] = 1;

    /* allocate CSR for M */
    M->rows = N;
    M->indptr = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(N+1));
    M->indptr[0] = 0;
    for(uint32_t i=0;i<N;++i) M->indptr[i+1] = M->indptr[i] + row_count[i];
    M->nnz = M->indptr[N];
    M->data = (double*)xaligned_malloc(sizeof(double)*M->nnz);
    M->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*M->nnz);

    /* position cursors per row */
    uint32_t *pos = (uint32_t*)calloc(N, sizeof(uint32_t));
    if(!pos){ fprintf(stderr, "OOM pos\n"); exit(1); }

    /* fill entries: for each column j, put contributions for rows r in band */
    for(uint32_t j=0;j<N;++j){
        uint32_t lo = (j > bw_spai) ? j - bw_spai : 0;
        uint32_t hi = (j + bw_spai < N-1) ? j + bw_spai : N-1;
        for(uint32_t k=jc[j]; k<jc[j+1]; ++k){
            uint32_t r = ir[k];
            if(r < lo || r > hi) continue;
            uint32_t idx = M->indptr[r] + pos[r]++;
            M->indices[idx] = j;
            if(r == j){
                M->data[idx] = 1.0 / diag[r];           /* diagonal */
            } else {
                M->data[idx] = - pr[k] / (diag[r] * diag[j]); /* -A_rj/(D_rr*D_jj) */
            }
        }
    }

    /* safety: if any row ended up empty (shouldn't after guard), put diagonal */
    for(uint32_t i=0;i<N;++i){
        if(M->indptr[i] == M->indptr[i+1]){
            uint32_t idx = M->indptr[i];
            M->indices[idx] = i;
            M->data[idx] = 1.0 / diag[i];
            pos[i] = 1;
        }
    }

    free(row_count);
    free(pos);
    free(diag);
}

static int solve_dense_pivot(double *A, double *b, int n){
    // Solve A x = b in-place (A overwritten) using Gaussian elimination with partial pivoting
    // A is column-major or row-major? We'll store A as row-major (A[i*n + j])
    const double EPS = 1e-18;
    for(int k=0;k<n;k++){
    // find pivot
    int piv = k; double maxv = fabs(A[k*n + k]);
    for(int i=k+1;i<n;i++){ double v = fabs(A[i*n + k]); if(v > maxv){ maxv = v; piv = i; } }
    if(maxv < EPS) return -1; // singular
    if(piv != k){
    // swap rows piv and k in A and b
    for(int j=k;j<n;j++){ double t = A[k*n + j]; A[k*n + j] = A[piv*n + j]; A[piv*n + j] = t; }
    double tb = b[k]; b[k] = b[piv]; b[piv] = tb;
    }
    // eliminate
    double akk = A[k*n + k];
    for(int i=k+1;i<n;i++){
    double fac = A[i*n + k] / akk;
    if(fac == 0.0) continue;
    for(int j=k;j<n;j++) A[i*n + j] -= fac * A[k*n + j];
    b[i] -= fac * b[k];
    }
    }
    // back substitution
        for(int i=n-1;i>=0;i--){
        double s = b[i];
        for(int j=i+1;j<n;j++) s -= A[i*n + j]*b[j];
        double diag = A[i*n + i]; if(fabs(diag) < 1e-30) return -1;
        b[i] = s/diag;
        }
    return 0;
}

/* Вспомогательная QR (как раньше) - решает min ||A x - b|| путем QR (A m x n, m >= n)
   A - column-major (col*m + row). Перезаписывает A и b. Возвращает 0 при успехе. */
static int qr_least_squares_householder(double *A, int m, int n, double *b, double *x){
    if(m < 1 || n < 1) return 1;
    for(int k=0;k<n;++k){
        /* norm of column k from row k..m-1 */
        double norm = 0.0;
        for(int i=k;i<m;++i){
            double v = A[(size_t)k*(size_t)m + i];
            norm += v*v;
        }
        norm = sqrt(norm);
        if(norm == 0.0) continue;
        double akk = A[(size_t)k*(size_t)m + k];
        double alpha = (akk >= 0.0) ? -norm : norm;
        double v0 = akk - alpha;
        A[(size_t)k*(size_t)m + k] = alpha;
        double vsq = v0*v0;
        for(int i=k+1;i<m;++i) vsq += A[(size_t)k*(size_t)m + i]*A[(size_t)k*(size_t)m + i];
        if(vsq == 0.0) continue;
        for(int j=k;j<n;++j){
            double dot = v0 * A[(size_t)j*(size_t)m + k];
            for(int i=k+1;i<m;++i) dot += A[(size_t)k*(size_t)m + i] * A[(size_t)j*(size_t)m + i];
            double tau = 2.0 * dot / vsq;
            A[(size_t)j*(size_t)m + k] -= tau * v0;
            for(int i=k+1;i<m;++i) A[(size_t)j*(size_t)m + i] -= tau * A[(size_t)k*(size_t)m + i];
        }
        double dotb = v0 * b[k];
        for(int i=k+1;i<m;++i) dotb += A[(size_t)k*(size_t)m + i] * b[i];
        double taub = 2.0 * dotb / vsq;
        b[k] -= taub * v0;
        for(int i=k+1;i<m;++i) b[i] -= taub * A[(size_t)k*(size_t)m + i];
    }
    /* back substitution R x = b[0..n-1] */
    for(int i=0;i<n;++i) x[i] = 0.0;
    for(int row = n-1; row >= 0; --row){
        double s = b[row];
        for(int col = row+1; col < n; ++col) s -= A[(size_t)col*(size_t)m + row] * x[col];
        double rdiag = A[(size_t)row*(size_t)m + row];
        if(fabs(rdiag) < 1e-18) return 1;
        x[row] = s / rdiag;
    }
    return 0;
}

/* helper: comparator for qsort (descending by absolute value) */
typedef struct { double val; int idx; } paird;
static int cmp_paird(const void* a, const void* b){
    double va = fabs(((const paird*)a)->val);
    double vb = fabs(((const paird*)b)->val);
    if(va < vb) return 1;
    if(va > vb) return -1;
    return 0;
}

/* main improved SPAI */
static void spai_build_improved(const mat_sparse_t* A, uint32_t N, CSRMatrix* M){
    const uint32_t* jc = (const uint32_t*)A->jc;
    const uint32_t* ir = (const uint32_t*)A->ir;
    const double* pr = (const double*)A->data;

    /* Параметры (попробуйте менять) */
    const uint32_t bw_spai = 16;      /* окно +/- по столбцам (48..128) */
    const int max_s = 128;            /* ограничение на число столбцов в локальной задаче */
    const int keep_k = 24;            /* сохраняем топ-k элементов в каждой строке */
    double reg_scale = 1e-4;          /* базовая регуляризация */
    double keep_tol = 1e-8;           /* минимальный модуль допуска (дополнительно к top-k) */
    double clip_scale = 1e3;          /* ограничение по абсолютной величине относительно 1/diag */

    /* диагональ на fallback */
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

    uint32_t *row_count = (uint32_t*)calloc(N, sizeof(uint32_t));
    if(!row_count){ fprintf(stderr,"OOM row_count\n"); exit(1); }

    /* FIRST PASS: только считаем, сколько nnz будет (подготовка CSR) */
    for(uint32_t i=0;i<N;++i){
        uint32_t lo = (i > bw_spai) ? i - bw_spai : 0;
        uint32_t hi = (i + bw_spai < N-1) ? i + bw_spai : N-1;
        int s = (int)(hi - lo + 1);
        if(s > max_s) s = max_s; /* ограничим */

        /* собираем список уникальных строк, где есть значения в колонках шаблона */
        int *row_map = (int*)malloc(sizeof(int)*N);
        if(!row_map){ fprintf(stderr,"OOM row_map\n"); exit(1); }
        for(uint32_t t=0;t<N;++t) row_map[t] = -1;
        int *rows = (int*)malloc(sizeof(int)*(size_t)(s * 8 + 16));
        if(!rows){ fprintf(stderr,"OOM rows\n"); exit(1); }
        int mrows = 0;

        /* если s == max_s, мы берем первые s столбцов lo..lo+s-1 (ближайшие) */
        int actual_s = 0;
        for(int p=0; p<s; ++p){
            uint32_t col = lo + p;
            actual_s++;
            for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
                uint32_t r = ir[k];
                if(row_map[r] == -1){
                    row_map[r] = mrows;
                    rows[mrows++] = (int)r;
                }
            }
        }
        /* гарантируем, что строка i присутствует */
        if(row_map[i] == -1){
            row_map[i] = mrows; rows[mrows++] = (int)i;
        }

        /* подготовим B (mrows x actual_s) и rhs (mrows), но только для подсчёта nonzero: решать не надо полностью */
        if(mrows <= 0){
            row_count[i] = 1;
            free(row_map); free(rows);
            continue;
        }

        /* малый dense буфер */
        double *B = (double*)calloc((size_t)mrows * (size_t)actual_s, sizeof(double));
        double *rhs = (double*)calloc((size_t)mrows, sizeof(double));
        double *sol = (double*)calloc((size_t)actual_s, sizeof(double));
        if(!B || !rhs || !sol){ fprintf(stderr,"OOM dense\n"); exit(1); }

        for(int p=0;p<actual_s;++p){
            uint32_t col = lo + p;
            for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
                uint32_t r = ir[k];
                int rp = row_map[r];
                if(rp >= 0) B[(size_t)p*(size_t)mrows + (size_t)rp] = pr[k];
            }
        }
        rhs[row_map[i]] = 1.0;

        /* регуляризация Tikhonov: добавим явное sqrt(lambda)*I в конец матрицы (в pass 1 только оценка) */
        double fro = 0.0;
        for(size_t jj=0;jj<(size_t)mrows*(size_t)actual_s;++jj) fro += B[jj]*B[jj];
        double lambda = reg_scale * ( (fro > 0.0) ? fro / (double)(mrows*actual_s) : 1e-8 );
        if(lambda < 1e-12) lambda = 1e-12;

        /* решаем (быстро) и считаем сколько ненулевых (по top-k) */
        /* создадим B_aug (mrows+actual_s) x actual_s и rhs_aug */
        int m_aug = mrows + actual_s;
        double *B_aug = (double*)calloc((size_t)m_aug * (size_t)actual_s, sizeof(double));
        double *rhs_aug = (double*)calloc((size_t)m_aug, sizeof(double));
        if(!B_aug || !rhs_aug){ fprintf(stderr,"OOM aug\n"); exit(1); }
        /* copy B */
        for(int p=0;p<actual_s;++p){
            for(int rpos=0;rpos<mrows;++rpos) B_aug[(size_t)p*(size_t)m_aug + (size_t)rpos] = B[(size_t)p*(size_t)mrows + (size_t)rpos];
            /* add reg row at position mrows + p */
            B_aug[(size_t)p*(size_t)m_aug + (size_t)(mrows + p)] = sqrt(lambda);
        }
        for(int r=0;r<mrows;++r) rhs_aug[r] = rhs[r];
        for(int r=mrows;r<m_aug;++r) rhs_aug[r] = 0.0;

        int ok = qr_least_squares_householder(B_aug, m_aug, actual_s, rhs_aug, sol);

        int written = 0;
        if(ok == 0){
            /* pick top-k by abs */
            paird *arr = (paird*)malloc(sizeof(paird)*(size_t)actual_s);
            if(!arr){ fprintf(stderr,"OOM arr\n"); exit(1); }
            for(int p=0;p<actual_s;++p){ arr[p].val = sol[p]; arr[p].idx = p; }
            qsort(arr, (size_t)actual_s, sizeof(paird), cmp_paird);
            int kkeep = keep_k < actual_s ? keep_k : actual_s;
            for(int t=0;t<kkeep;++t){
                if(fabs(arr[t].val) <= keep_tol) break;
                ++written;
            }
            free(arr);
        }
        if(written == 0) written = 1; /* всегда минимум - диагональ */

        row_count[i] = (uint32_t)written;

        free(B); free(rhs); free(sol); free(B_aug); free(rhs_aug); free(row_map); free(rows);
    } /* end first pass */

    /* allocate CSR */
    M->rows = N;
    M->indptr = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(N+1));
    if(!M->indptr){ fprintf(stderr,"OOM indptr\n"); exit(1); }
    M->indptr[0] = 0;
    for(uint32_t i=0;i<N;++i) M->indptr[i+1] = M->indptr[i] + row_count[i];
    M->nnz = M->indptr[N];
    M->data = (double*)xaligned_malloc(sizeof(double)*(size_t)M->nnz);
    M->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(size_t)M->nnz);
    if((!M->data) || (!M->indices)){ fprintf(stderr,"OOM data/indices\n"); exit(1); }

    uint32_t *pos = (uint32_t*)calloc(N, sizeof(uint32_t));
    if(!pos){ fprintf(stderr,"OOM pos\n"); exit(1); }

    /* SECOND PASS: recompute локальные решения и записать (параллельно) */
    #pragma omp parallel for schedule(guided)
    for(int32_t ii=0; ii<(int32_t)N; ++ii){
        uint32_t i = (uint32_t)ii;
        uint32_t lo = (i > bw_spai) ? i - bw_spai : 0;
        uint32_t hi = (i + bw_spai < N-1) ? i + bw_spai : N-1;
        int s = (int)(hi - lo + 1);
        if(s > max_s) s = max_s;

        int *row_map = (int*)malloc(sizeof(int)*N);
        if(!row_map){ fprintf(stderr,"OOM row_map2\n"); exit(1); }
        for(uint32_t t=0;t<N;++t) row_map[t] = -1;
        int *rows = (int*)malloc(sizeof(int)*(size_t)(s * 8 + 16));
        if(!rows){ fprintf(stderr,"OOM rows2\n"); exit(1); }
        int mrows = 0;
        for(int p=0;p<s;++p){
            uint32_t col = lo + p;
            for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
                uint32_t r = ir[k];
                if(row_map[r] == -1){
                    row_map[r] = mrows;
                    rows[mrows++] = (int)r;
                }
            }
        }
        if(row_map[i] == -1){
            row_map[i] = mrows; rows[mrows++] = (int)i;
        }

        if(mrows <= 0){
            uint32_t idx = M->indptr[i] + __atomic_fetch_add(&pos[i], 1, __ATOMIC_RELAXED);
            M->indices[idx] = i; M->data[idx] = 1.0/diag[i];
            free(row_map); free(rows); continue;
        }

        double *B = (double*)calloc((size_t)mrows * (size_t)s, sizeof(double));
        double *rhs = (double*)calloc((size_t)mrows, sizeof(double));
        double *sol = (double*)calloc((size_t)s, sizeof(double));
        if(!B || !rhs || !sol){ fprintf(stderr,"OOM dense2\n"); exit(1); }

        for(int p=0;p<s;++p){
            uint32_t col = lo + p;
            for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
                uint32_t r = ir[k];
                int rp = row_map[r];
                if(rp >= 0) B[(size_t)p*(size_t)mrows + (size_t)rp] = pr[k];
            }
        }
        rhs[row_map[i]] = 1.0;

        double fro = 0.0;
        for(size_t jj=0;jj<(size_t)mrows*(size_t)s;++jj) fro += B[jj]*B[jj];
        double lambda = reg_scale * ( (fro > 0.0) ? fro / (double)(mrows*s) : 1e-8 );
        if(lambda < 1e-12) lambda = 1e-12;

        int m_aug = mrows + s;
        double *B_aug = (double*)calloc((size_t)m_aug * (size_t)s, sizeof(double));
        double *rhs_aug = (double*)calloc((size_t)m_aug, sizeof(double));
        if(!B_aug || !rhs_aug){ fprintf(stderr,"OOM aug2\n"); exit(1); }
        for(int p=0;p<s;++p){
            for(int rpos=0;rpos<mrows;++rpos) B_aug[(size_t)p*(size_t)m_aug + (size_t)rpos] = B[(size_t)p*(size_t)mrows + (size_t)rpos];
            B_aug[(size_t)p*(size_t)m_aug + (size_t)(mrows + p)] = sqrt(lambda);
        }
        for(int r=0;r<mrows;++r) rhs_aug[r] = rhs[r];
        for(int r=mrows;r<m_aug;++r) rhs_aug[r] = 0.0;

        int ok = qr_least_squares_householder(B_aug, m_aug, s, rhs_aug, sol);

        int local_written = 0;
        if(ok == 0){
            paird *arr = (paird*)malloc(sizeof(paird)*(size_t)s);
            if(!arr){ fprintf(stderr,"OOM arr2\n"); exit(1); }
            for(int p=0;p<s;++p){ arr[p].val = sol[p]; arr[p].idx = p; }
            qsort(arr, (size_t)s, sizeof(paird), cmp_paird);
            int kkeep = keep_k < s ? keep_k : s;
            for(int t=0;t<kkeep;++t){
                double val = arr[t].val;
                if(fabs(val) <= keep_tol) break;
                /* clip relative to diagonal inverse magnitude to avoid runaways */
                double clip = clip_scale * fabs(1.0 / diag[lo + arr[t].idx]);
                if(fabs(val) > clip) val = (val > 0.0) ? clip : -clip;
                uint32_t idx = M->indptr[i] + __atomic_fetch_add(&pos[i], 1, __ATOMIC_RELAXED);
                M->indices[idx] = lo + arr[t].idx;
                M->data[idx] = val;
                ++local_written;
            }
            free(arr);
        }
        if(local_written == 0){
            uint32_t idx = M->indptr[i] + __atomic_fetch_add(&pos[i], 1, __ATOMIC_RELAXED);
            M->indices[idx] = i;
            M->data[idx] = 1.0/diag[i];
        }

        free(B); free(rhs); free(sol); free(B_aug); free(rhs_aug); free(row_map); free(rows);
    } /* end parallel */

    free(pos); free(row_count); free(diag);
}


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
