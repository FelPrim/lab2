#include <stdio.h>
#include <matio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#if defined(_WIN32)
#include <malloc.h>
static inline void *aligned_alloc( size_t alignment, size_t size){
    return _aligned_malloc(size, alignment);
}
#endif

typedef struct Vector{
    alignas(32) double *ptr;
    uint32_t capacity;
    uint32_t size;
} Vector;

typedef struct CSRMatrix{
    Vector data;
    alignas(32) uint32_t *indices;
    uint32_t *indptr;
} CSRMatrix;

// preconditioner
int spai_build_band_csr(
    const mat_sparse_t *A,
    uint32_t N,
    int bw,
    int s_max,
    CSRMatrix *M_out
);

// y = M * x
void spai_apply(
    const CSRMatrix *M,
    const double * restrict x,
    double * restrict y
);

// бинарный поиск A(j,k) в CSR-строке j
double get_A_jk(
    const uint32_t *ir, const uint32_t *jc, const double *ax,
    uint32_t k, uint32_t j);

// скалярное произведение двух CSR-строк (merge двух отсортированных списков)
double row_dot(
    const uint32_t *ir, const uint32_t *jc, const double *ax,
    uint32_t kc, uint32_t kd);

int chol_spd_inplace(double *G, int n);

void chol_solve_LLt(const double *L, double *x, const double *b, int n);

static inline int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

double TOLERANCE = 1e-9;
unsigned int MAXITER = 10000;
unsigned int THREADS_NUM = 2;
unsigned int SMALLCACHE_SZ = 32000;
unsigned int BIGCACHE_SZ = 512000;

int main(int argc, char *argv[]){
    char* FILENAME = "pwtk.mat";
    char* VARNAME = "Problem";
    char* FIELDNAME = "A";
    if (argc == 1){
    }
    else if (argc == 4){
        FILENAME = argv[1];
        VARNAME = argv[2];
        FIELDNAME = argv[3];
    }
    else if (argc == 9){
        FILENAME = argv[1];
        VARNAME = argv[2];
        FIELDNAME = argv[3];
        TOLERANCE = strtod(argv[4], (void*)NULL);
        MAXITER = strtoul(argv[5], (void*) NULL, 10);
        THREADS_NUM = strtoul(argv[6], (void*) NULL, 10);
        SMALLCACHE_SZ = strtoul(argv[7], (void*) NULL, 10);
        BIGCACHE_SZ = strtoul(argv[8], (void*) NULL, 10);
    }
    else
        printf("This program can't handle %d arguements - only 1|4|9 are supported.\n", argc);
    
    mat_t *file = Mat_Open(FILENAME, MAT_ACC_RDONLY);
    if (!file)
        goto NO_FILE;
    matvar_t *var = Mat_VarRead(file, VARNAME);
    Mat_Close(file);
    if (!var)
        goto NO_VAR;
    if (var->class_type != MAT_C_STRUCT)
        goto NO_STRUCT;
    matvar_t *field = Mat_VarGetStructFieldByName(var, FIELDNAME, 0);
    if (!field)
        goto NO_FIELD;
    if (field->class_type != MAT_C_SPARSE || field->data_type != MAT_T_DOUBLE || field->data == NULL)
        goto NO_MATRIX;
    Mat_VarPrint(field, 0);
    mat_sparse_t *input = field->data;
    printf("nzmax=%u, nir=%u, njc=%u, ndata=%u\n",
    input->nzmax, input->nir, input->njc, input->ndata);
    printf("First few jc:");
    for (int t = 0; t < 5 && t < field->dims[1]+1; ++t){
        printf("%u ", input->jc[t]);
    }

    

    return 0;

NO_FILE:
    puts("No file has been found");
    return 1;

NO_VAR:
    puts("No variable has been found");
    return 2;

NO_STRUCT:
    puts("No struct has been found");
    Mat_VarFree(var);
    return 3;

NO_FIELD:
    puts("No field has been found");
    return 4;

NO_MATRIX:
    puts("Matrix is wrong");
    Mat_VarFree(field);
    return 5;
    
}

// ПРЕДОБУСЛАВЛИВАТЕЛЬ

static inline double get_A_jk(
    const uint32_t *ir, const uint32_t *jc, const double *ax,
    uint32_t k, uint32_t j)
{
    uint32_t lo = jc[j], hi = jc[j+1];
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        uint32_t r = ir[mid];
        if (r == k) return ax[mid];
        if (r < k) lo = mid + 1; else hi = mid;
    }
    return 0.0;
}

double row_dot(
    const uint32_t *ir, const uint32_t *jc, const double *ax,
    uint32_t kc, uint32_t kd)
{
    uint32_t p = jc[kc], pe = jc[kc+1];
    uint32_t q = jc[kd], qe = jc[kd+1];
    double sum = 0.0;
    while (p < pe && q < qe) {
        uint32_t i = ir[p], j = ir[q];
        if (i == j) { sum += ax[p++] * ax[q++]; }
        else if (i < j) { p++; }
        else { q++; }
    }
    return sum;
}

static int chol_spd_inplace(double* G, int n)
{
    for (int j = 0; j < n; ++j) {
        double* Gj = G + j*n;
        // диагональ
        double sum = Gj[j];
        for (int k = 0; k < j; ++k) {
            double L_jk = Gj[k];
            sum -= L_jk * L_jk;
        }
        if (sum <= 0.0) return -1; 
        double L_jj = sqrt(sum);
        Gj[j] = L_jj;
        for (int i = j+1; i < n; ++i) {
            double* Gi = G + i*n;
            double s = Gi[j];
            #pragma omp simd reduction(-:s)
            for (int k = 0; k < j; ++k)
                s -= Gi[k] * Gj[k];
            Gi[j] = s / L_jj;
        }
    }
    for (int r = 0; r < n; ++r)
        for (int c = r+1; c < n; ++c)
            G[r*n + c] = 0.0;
    return 0;
}


static void chol_solve_LLt(const double* L, double* x, const double* b, int n)
{
    double* y = x;
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        #pragma omp simd reduction(-:s)
        for (int k = 0; k < i; ++k)
            s -= L[i*n + k] * y[k];
        y[i] = s / L[i*n + i];
    }
    for (int i = n-1; i >= 0; --i) {
        double s = y[i];
        for (int k = i+1; k < n; ++k) s -= L[k*n + i] * x[k];
        x[i] = s / L[i*n + i];
    }
}

int spai_build_band_csr(
    const mat_sparse_t* A,
    uint32_t N,
    int bw,
    int s_max,
    CSRMatrix* M_out)
{
    if (!A || !M_out || !A->ir || !A->jc || !A->data) return -1;
    const uint32_t *ir = A->ir;
    const uint32_t *jc = A->jc;
    const double   *ax = (const double*)A->data;

    if (bw < 0) bw = 0;
    if (s_max <= 0) s_max = 48;

    uint32_t* jcM = (uint32_t*)aligned_alloc(64, (size_t)(N+1) * sizeof(uint32_t));
    if (!jcM) return -2;
    jcM[0] = 0;
    for (uint32_t j = 0; j < N; ++j) {
        int lo = clampi((int)j - bw, 0, (int)N-1);
        int hi = clampi((int)j + bw, 0, (int)N-1);
        int s = hi - lo + 1;
        if (s > s_max) {
            int half = s_max/2;
            lo = clampi((int)j - half, 0, (int)N-1);
            hi = clampi(lo + s_max - 1, 0, (int)N-1);
            s = hi - lo + 1;
        }
        jcM[j+1] = jcM[j] + (uint32_t)s;
    }
    uint32_t nzM = jcM[N];

    uint32_t* irM = (uint32_t*)aligned_alloc(64, (size_t)nzM * sizeof(uint32_t));
    double*   xM  = (double*)  aligned_alloc(64, (size_t)nzM * sizeof(double));
    if (!irM || !xM) { free(jcM); free(irM); free(xM); return -3; }

    #pragma omp parallel for schedule(static)
    for (uint32_t j = 0; j < N; ++j) {
        uint32_t p = jcM[j];
        int lo = clampi((int)j - bw, 0, (int)N-1);
        int hi = clampi((int)j + bw, 0, (int)N-1);
        int s = hi - lo + 1;
        if (s > s_max) {
            int half = s_max/2;
            lo = clampi((int)j - half, 0, (int)N-1);
            hi = clampi(lo + s_max - 1, 0, (int)N-1);
            s = hi - lo + 1;
        }
        for (int t = 0; t < s; ++t) irM[p + t] = (uint32_t)(lo + t);
    }

    #pragma omp parallel
    {
        const int S_MAX = s_max;
        double* G = (double*)aligned_alloc(64, (size_t)S_MAX * (size_t)S_MAX * sizeof(double));
        double* r = (double*)aligned_alloc(64, (size_t)S_MAX * sizeof(double));
        double* x = (double*)aligned_alloc(64, (size_t)S_MAX * sizeof(double));
        if (!G || !r || !x) {
        }

        #pragma omp for schedule(static)
        for (uint32_t j = 0; j < N; ++j) {
            uint32_t p0 = jcM[j];
            uint32_t p1 = jcM[j+1];
            int s = (int)(p1 - p0);
            if (s <= 0) continue;
         // 3.1 r_t = A_{j, k_t}, k_t = irM[p0 + t]
            for (int t = 0; t < s; ++t) {
                uint32_t k = irM[p0 + t];
                r[t] = get_A_jk(ir, jc, ax, j, k);
            }
            for (int t = 0; t < s; ++t) {
                for (int u = 0; u <= t; ++u) {
                    double val = col_dot(ir, jc, ax, irM[p0 + t], irM[p0 + u]);
                    G[t*s + u] = val;
                    G[u*s + t] = val;
                }
            }
            int ok = chol_spd_inplace(G, s);
            if (ok == 0) {
                chol_solve_LLt(G, x, r, s);
            } else {
                for (int t = 0; t < s; ++t) {
                    double d = G[t*s + t];
                    x[t] = (fabs(d) > 1e-30) ? (r[t] / d) : 0.0;
                }
            }
            for (int t = 0; t < s; ++t) xM[p0 + t] = x[t];
        }

        if (G) free(G);
        if (r) free(r);
        if (x) free(x);
    } // omp parallel

    // 4) Выходная структура M_out
    M_out->nzmax = nzM;
    M_out->ir    = irM;  M_out->nir = nzM;
    M_out->jc    = jcM;  M_out->njc = N + 1;
    M_out->ndata = nzM;
    M_out->data  = xM;
    return 0;
}

// Применение y = M x
void spai_apply_csc(
    const mat_sparse_t* M,
    const double* __restrict x,
    double* __restrict y,
    uint32_t N)
{
    const uint32_t *ir = M->ir;
    const uint32_t *jc = M->jc;
    const double   *mx = (const double*)M->data;
    // y <- 0
    #pragma omp simd
    for (uint32_t i = 0; i < N; ++i) y[i] = 0.0;
    // y += sum_j x[j] * M[:,j]
    for (uint32_t j = 0; j < N; ++j) {
        double xj = x[j];
        uint32_t p = jc[j], pe = jc[j+1];
        for (; p < pe; ++p) {
            y[ ir[p] ] += mx[p] * xj;
        }
    }
}


