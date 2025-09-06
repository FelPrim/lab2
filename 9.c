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
#include <xmmintrin.h>

#define ALIGN 32
#define ALIGNED_ELEM (ALIGN/sizeof(double))
static_assert(ALIGN % sizeof(double) == 0);

//static double TOLERANCE = 1e-9;
static unsigned int MAXITER = 1000;
#ifndef MAXTHREADS
static unsigned int THREADS_NUM = 12;
#else
static unsigned int THREADS_NUM = MAXTHREADS;
#endif
#ifndef CACHE
static unsigned int BIGCACHE_SZ = 512000;
#else
static unsigned int BIGCACHE_SZ = CACHE;
#endif
static unsigned int BW = 6400;  

static double k = 7;
   
// C23!!!
static bool CONSOLE_OUTPUT = 1; 

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
    uint32_t *indptr; 
    uint32_t rows;
    uint32_t nnz;
} CSRMatrix;

// StripeMatrix == CSRMatrix + stripes information
typedef struct StripeMatrix{
    alignas(32) double *data;
    alignas(32) uint32_t *indices;
    uint32_t *indptr; // length rows+1
    uint32_t rows;
    uint32_t nnz;

    // stripes data: each entry stripes[i] is the first row index of stripe i
    // stripes_len is number of stripes; stripes[stripes_len] is implicitly rows
    uint32_t *stripes;
    uint32_t stripes_len;
} StripeMatrix;

static inline double wall_seconds(){ return omp_get_wtime(); }

static void stripe_free(StripeMatrix *A){
    if(!A) return;
    xaligned_free(A->data); 
    xaligned_free(A->indices); 
    xaligned_free(A->indptr);
    if(A->stripes) free(A->stripes);
    memset(A,0,sizeof(*A));
}

static void csr_free(CSRMatrix *A){
    if(!A) return;
    xaligned_free(A->data); 
    xaligned_free(A->indices); 
    xaligned_free(A->indptr);
    memset(A,0,sizeof(*A));
}

//static CSRMatrix csr_empty(uint32_t n, uint32_t nnz){
//    CSRMatrix A; memset(&A,0,sizeof(A));
//    A.rows = n; A.nnz = nnz;
//    A.data    = (double*)xaligned_malloc(sizeof(double)*nnz);
//    A.indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*nnz);
//    A.indptr  = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(n+1));
//    return A;
//}


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

///////////////////////////////////////////
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
    for(uint32_t i=0;i<n;++i)
		y[i] *= a;
}
static inline double vdot(const double* restrict a, const double* restrict b, uint32_t n){
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for(uint32_t i=0;i<n;++i) 
		sum += a[i]*b[i];
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
static uint32_t compute_band_threshold(){
    double val = (k*(double)BIGCACHE_SZ - 8.0*(double)BW)/12.0;
    if(val < 1.0) return 1u;
    return (uint32_t)floor(val);
}

static void compute_stripes_for_band(StripeMatrix *A){
    if(!A || A->rows==0){ A->stripes = NULL; A->stripes_len = 0; return; }
    uint32_t thresh = compute_band_threshold();
    // estimate number of stripes
   // uint32_t est = (A->nnz / (thresh>0?thresh:1)) + 1;
	uint32_t max_stripes = A->rows + 2;
    uint32_t *tmp = (uint32_t*)malloc(sizeof(uint32_t)*max_stripes);
    uint32_t s = 0;
    tmp[s++] = 0;
    uint32_t cur_nnz = 0;
    for(uint32_t i=0;i<A->rows;++i){
        uint32_t row_nnz = A->indptr[i+1] - A->indptr[i];
        if(cur_nnz + row_nnz > thresh && cur_nnz>0){
            // cut before row i
            tmp[s++] = i;
            cur_nnz = row_nnz;
        } else {
            cur_nnz += row_nnz;
        }
    }
    if(tmp[s-1] != A->rows) tmp[s++] = A->rows;
    // shrink
    A->stripes = (uint32_t*)malloc(sizeof(uint32_t)*s);
    memcpy(A->stripes, tmp, sizeof(uint32_t)*s);
    A->stripes_len = s-1; // number of stripes = entries-1
    free(tmp);
}

// For Other: accumulate rows until 12*N + 8*(M) < k*BIGCACHE_SZ
static void compute_stripes_for_other(StripeMatrix *A){
    if(!A || A->rows==0){ A->stripes=NULL; A->stripes_len=0; return; }
    double cap = k*(double)BIGCACHE_SZ;
    // allocate temporary ranges
    uint32_t *tmp = (uint32_t*)malloc(sizeof(uint32_t)*(A->rows+2));
    uint32_t s = 0; tmp[s++] = 0;
    uint32_t cur_nnz = 0;
    uint32_t cur_minj = UINT32_MAX, cur_maxj = 0;
    for(uint32_t i=0;i<A->rows;++i){
        uint32_t row_start = A->indptr[i];
        uint32_t row_end = A->indptr[i+1];
        uint32_t row_nnz = row_end - row_start;
        if(row_nnz==0){ // empty row -- just continue
            // but keep it in stripe
        } else {
            // find min and max column index in this row
            uint32_t rmin = UINT32_MAX, rmax = 0;
            for(uint32_t kidx=row_start; kidx<row_end; ++kidx){ uint32_t col = A->indices[kidx]; if(col<rmin) rmin = col; if(col>rmax) rmax = col; }
            if(cur_minj==UINT32_MAX){ cur_minj = rmin; cur_maxj = rmax; }
            else { if(rmin < cur_minj) cur_minj = rmin; if(rmax > cur_maxj) cur_maxj = rmax; }
        }
        uint64_t est_mem = (uint64_t)12*(uint64_t)(cur_nnz + row_nnz) + (uint64_t)8*( (cur_minj==UINT32_MAX?1:(cur_maxj - cur_minj + 1)) );
        if(cur_nnz>0 && est_mem > (uint64_t)cap){
            // cut before current row
            tmp[s++] = i;
            // start new stripe containing this row
            cur_nnz = row_nnz;
            if(row_nnz==0){ cur_minj = UINT32_MAX; cur_maxj = 0; }
            else { // recompute min/max for this row
                uint32_t rmin2 = UINT32_MAX, rmax2 = 0;
                for(uint32_t kidx=row_start; kidx<row_end; ++kidx){ uint32_t col = A->indices[kidx]; if(col<rmin2) rmin2 = col; if(col>rmax2) rmax2 = col; }
                cur_minj = rmin2; cur_maxj = rmax2;
            }
        } else {
            cur_nnz += row_nnz;
        }
    }
    if(tmp[s-1] != A->rows) tmp[s++] = A->rows;
    A->stripes = (uint32_t*)malloc(sizeof(uint32_t)*s);
    memcpy(A->stripes, tmp, sizeof(uint32_t)*s);
    A->stripes_len = s-1;
    free(tmp);
}

// Convert CSC (MATLAB sparse) to CSR-like storage and then compute stripes for Band and Other
static void csc_to_csr_and_split(const mat_sparse_t* A, uint32_t N, StripeMatrix* Band, StripeMatrix* Other){

    const uint32_t* jc = (const uint32_t*)A->jc;
    const uint32_t* ir = (const uint32_t*)A->ir;
    const double*  pr = (const double*) A->data;
   // const uint32_t nnz = (uint32_t)A->ndata;

    // first count nnz per row for band/other
    uint32_t *row_band = (uint32_t*)calloc(N,sizeof(uint32_t));
    uint32_t *row_other= (uint32_t*)calloc(N,sizeof(uint32_t));
    for(uint32_t col=0; col<N; ++col){
        for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
            uint32_t r = ir[k];
            if((int)abs((int)r-(int)col) <= (int)BW) row_band[r]++; else row_other[r]++;
        }
    }
    
    memset(Band, 0, sizeof(StripeMatrix));
    memset(Other, 0, sizeof(StripeMatrix));
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
    Band->data    = (double*)xaligned_malloc(sizeof(double)*(size_t)Band->nnz);
    Band->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(size_t)Band->nnz);
    Other->data    = (double*)xaligned_malloc(sizeof(double)*(size_t)Other->nnz);
    Other->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(size_t)Other->nnz);

    uint32_t *wb = (uint32_t*)calloc(N,sizeof(uint32_t));
    uint32_t *wo = (uint32_t*)calloc(N,sizeof(uint32_t));
    

    for(uint32_t col=0; col<N; ++col){
        for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
            uint32_t r = ir[k];
            if((int)abs((int)r-(int)col) <= (int)BW){
                uint32_t pos = Band->indptr[r] + wb[r]++;
                Band->indices[pos] = col; Band->data[pos] = pr[k];
            }else{
                uint32_t pos = Other->indptr[r] + wo[r]++;
                Other->indices[pos] = col; Other->data[pos] = pr[k];
            }
        }
    }
    free(wb); free(wo); free(row_band); free(row_other);

    // compute stripes according to cache model
    compute_stripes_for_band(Band);
    compute_stripes_for_other(Other);

    // print summary
	if (CONSOLE_OUTPUT){
		printf("Split completed: Band nnz=%u, Other nnz=%u\n", Band->nnz, Other->nnz);
		printf("Band stripes: %u, Other stripes: %u\n", Band->stripes_len, Other->stripes_len);
		// compute per-stripe nnz stats
		if(Band->stripes_len){
			uint32_t minn = UINT32_MAX, maxn = 0;
			for(uint32_t s=0;s<Band->stripes_len;++s){ uint32_t a=Band->stripes[s], b=Band->stripes[s+1]; uint32_t cnt = Band->indptr[b]-Band->indptr[a]; if(cnt<minn) minn=cnt; if(cnt>maxn) maxn=cnt; }
			printf("Band stripe nnz min=%u max=%u\n", minn==UINT32_MAX?0:minn, maxn);
		}
		
		if(Other->stripes_len){
			uint32_t minn = UINT32_MAX, maxn = 0;
			for(uint32_t s=0;s<Other->stripes_len;++s){ uint32_t a=Other->stripes[s], b=Other->stripes[s+1]; uint32_t cnt = Other->indptr[b]-Other->indptr[a]; if(cnt<minn) minn=cnt; if(cnt>maxn) maxn=cnt; }
			printf("Other stripe nnz min=%u max=%u\n", minn==UINT32_MAX?0:minn, maxn);
		}
	}
}

static void csc_to_csr_single(const mat_sparse_t* A, uint32_t N, CSRMatrix* Out){
	const uint32_t* jc = (const uint32_t*)A->jc;
	const uint32_t* ir = (const uint32_t*)A->ir;
	const double* pr = (const double*) A->data;
	const uint32_t nnz = (uint32_t)A->ndata;
	
	memset(Out, 0, sizeof(CSRMatrix));
	Out->rows = N;
	Out->nnz = nnz;
	Out->indptr = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(N+1));
	Out->data = (double*)xaligned_malloc(sizeof(double)*(size_t)nnz);
	Out->indices = (uint32_t*)xaligned_malloc(sizeof(uint32_t)*(size_t)nnz);

	// count nnz per row
	uint32_t *row_counts = (uint32_t*)calloc(N,sizeof(uint32_t));
	for(uint32_t col=0; col<N; ++col){
		for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
			uint32_t r = ir[k];
			row_counts[r]++;
		}
	}

	Out->indptr[0] = 0;
	for(uint32_t i=0;i<N;++i) 
		Out->indptr[i+1] = Out->indptr[i] + row_counts[i];

	// fill
	uint32_t *w = (uint32_t*)calloc(N,sizeof(uint32_t));
	for(uint32_t col=0; col<N; ++col){
		for(uint32_t k=jc[col]; k<jc[col+1]; ++k){
			uint32_t r = ir[k];
			uint32_t pos = Out->indptr[r] + w[r]++;
			Out->indices[pos] = col;
			Out->data[pos] = pr[k];
		}
	}
	free(w); 
	free(row_counts);
	printf("CSC->CSR single conversion completed: rows=%u nnz=%u\n", N, nnz);
}

// ===================== SPAI(0) builder (diagonal-only variant) =====================
// Build diagonal inverse into Vector M: M[i] = 1.0 / A[i,i]
static void spai0_build_diagonal(const mat_sparse_t* A, uint32_t N, Vector* M){
    const uint32_t* jc = (const uint32_t*)A->jc; // size N+1
    const uint32_t* ir = (const uint32_t*)A->ir; 
    const double*  pr = (const double*) A->data;

    *M = vec_empty(N);
    for(uint32_t i=0;i<N;++i) M->ptr[i] = 1.0; // fallback
    for(uint32_t col=0; col<N; ++col){
        for(uint32_t k=jc[col]; k<jc[col+1]; ++k){ 
			uint32_t r = ir[k]; 
			if(r==col){ 
				double aii = pr[k];
				M->ptr[col] = (fabs(aii)>0.0)? 1.0/aii : 1.0;
				break;
			} 
		}
    }
}

static bool save_vector_binary(const char* path, const Vector* v){
    FILE *f = fopen(path, "wb"); if(!f){ perror("fopen save"); return false; }
    if(fwrite(&v->size, sizeof(uint32_t), 1, f) != 1){ fclose(f); return false; }
    if(fwrite(v->ptr, sizeof(double), v->size, f) != v->size){ fclose(f); return false; }
    fclose(f); return true;
}
static bool load_vector_binary(const char* path, Vector* v){
    FILE *f = fopen(path, "rb"); if(!f) return false;
    uint32_t n=0; if(fread(&n,sizeof(uint32_t),1,f)!=1){ fclose(f); return false; }
    *v = vec_empty(n);
    if(fread(v->ptr, sizeof(double), n, f) != n){ vec_free(v); fclose(f); return false; }
    fclose(f); return true;
}

// ===================== SpMV over stripes =====================
// y = A * x (full apply)
// static void stripe_apply(const StripeMatrix* A, const double* restrict x, double* restrict y){
//     if(!A || A->nnz==0) return;
//     // iterate over stripes in parallel
//     uint32_t S = A->stripes_len;
//     #pragma omp parallel for schedule(static)
//     for(uint32_t s=0; s<S; ++s){
//         uint32_t rstart = A->stripes[s];
//         uint32_t rend   = A->stripes[s+1];
//         for(uint32_t i=rstart; i<rend; ++i){
//             double sum = 0.0;
//             for(uint32_t k=A->indptr[i]; k<A->indptr[i+1]; ++k) 
// 				sum += A->data[k]*x[A->indices[k]];
//             y[i] = sum;
//         }
//     }
// }
// 
// // y += A*x : iterate over stripes and add
// static void stripe_apply_add(const StripeMatrix* A, const double* restrict x, double* restrict y){
//     if(!A || A->nnz==0) return;
//     uint32_t S = A->stripes_len;
//     #pragma omp parallel for schedule(static)
//     for(uint32_t s=0; s<S; ++s){
//         uint32_t rstart = A->stripes[s];
//         uint32_t rend   = A->stripes[s+1];
//         for(uint32_t i=rstart; i<rend; ++i){
//             double sum = 0.0;
//             for(uint32_t k=A->indptr[i]; k<A->indptr[i+1]; ++k)
// 				sum += A->data[k]*x[A->indices[k]];
//             y[i] += sum;
//         }
//     }
// }
// 

static void apply_both(const StripeMatrix* Band, const StripeMatrix* Other, const double* restrict x, double* restrict y){
	memset(y, 0, Band->rows*sizeof(double));
	//stripe_apply(Band, x, y);
	//stripe_apply_add(Other, x, y);
	
	uint32_t S = Band->stripes_len + Other->stripes_len;
	#pragma omp parallel for schedule(guided)
    for(uint32_t s=0; s<S; ++s){
		if (s < Band->stripes_len){
			uint32_t rstart = Band->stripes[s];
			_mm_prefetch(&x[Band->indices[Band->indptr[rstart]]], _MM_HINT_T1);
			uint32_t rend   = Band->stripes[s+1];
			for(uint32_t i=rstart; i<rend; ++i){
				double sum = 0.0;
				for(uint32_t k=Band->indptr[i]; k<Band->indptr[i+1]; ++k)
					sum += Band->data[k]*x[Band->indices[k]];
				y[i] += sum;
			}
		}
		else {
			// s-Band->stripes_len < Other->stripes_len
			uint32_t real_s = s - Band->stripes_len;
			uint32_t rstart = Other->stripes[real_s];
			uint32_t rend   = Other->stripes[real_s+1];
			for(uint32_t i=rstart; i<rend; ++i){
				double sum = 0.0;
				for(uint32_t k=Other->indptr[i]; k<Other->indptr[i+1]; ++k)
					sum += Other->data[k]*x[Other->indices[k]];
				y[i] += sum;
			}
		}			
    }
}

static void csr_apply(const CSRMatrix* A, const double* restrict x, double* restrict y){
    const uint32_t n = A->rows; 
	memset(y, 0, n*sizeof(double));
    #pragma omp parallel for schedule(guided)
    for(uint32_t i=0;i<n;++i){
        double sum=0.0;
        for(uint32_t k=A->indptr[i]; k<A->indptr[i+1]; ++k){ 
			sum += A->data[k]*x[A->indices[k]];		
		}
        y[i]=sum;
    }
}

// Apply diagonal preconditioner stored in Vector: z = M .* r
static void apply_precond(const Vector* M, const double* restrict r, double* restrict z){
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<M->size;++i) 
		z[i] = M->ptr[i] * r[i];
}

void compute_print_true_residual(const StripeMatrix* Band, const StripeMatrix* Other,
                                 const Vector* b, const Vector* x){
    uint32_t n = b->size;
    Vector Ax = vec_empty(n);
	printf("%u, %u\n", Band->rows, n);
	apply_both(Band, Other, x->ptr, Ax.ptr);
    // r_true = b - Ax  (reuse Ax to store residual)
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) 
		Ax.ptr[i] = b->ptr[i] - Ax.ptr[i];
    double tr = vnorm2(Ax.ptr, n);
    double bn = fmax(1e-30, vnorm2(b->ptr,n));
    printf("TRUE relative residual: %.6e\n", tr/bn);
    vec_free(&Ax);
}

void compute_print_true_residual_single(const CSRMatrix* A, const Vector* b, const Vector* x){
    uint32_t n = b->size;
    Vector Ax = vec_empty(n);
	csr_apply(A, x->ptr, Ax.ptr);
    // r_true = b - Ax  (reuse Ax to store residual)
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i) 
		Ax.ptr[i] = b->ptr[i] - Ax.ptr[i];
    double tr = vnorm2(Ax.ptr, n);
    double bn = fmax(1e-30, vnorm2(b->ptr,n));
    printf("TRUE relative residual: %.6e\n", tr/bn);
    vec_free(&Ax);
}

void check_AM_is_I(const StripeMatrix *Band, const StripeMatrix *Other, const Vector *Minv, uint32_t N){
    Vector e = vec_empty(N), tmp = vec_empty(N), out = vec_empty(N);
    srand(123);
    int NS = 10;
    double maxerr = 0.0;
    for(int s=0;s<NS;++s){
        int j = rand() % N;
        vzero(e.ptr, N); 
		e.ptr[j]=1.0;
        /* tmp = Minv * e */
        apply_precond(Minv, e.ptr, tmp.ptr);
        /* out = A * tmp */
		apply_both(Band, Other, tmp.ptr, out.ptr);
        out.ptr[j] -= 1.0; /* A*(M*e_j) - e_j */
        double err = vnorm2(out.ptr, N);
        printf("AM-I test j=%d err = %.6e\n", j, err);
        maxerr = fmax(maxerr, err);
    }
    printf("AM-I max(sample) = %.6e\n", maxerr);
    vec_free(&e); vec_free(&tmp); vec_free(&out);
}

// ===================== BiCGStab (left preconditioning with SPAI inverse) =====================
static int bicgstab_solve(const StripeMatrix* Band, const StripeMatrix* Other, const Vector* Minv,
                          const Vector* b, Vector* x, unsigned int* iters, double* out_final_res){
    
    const uint32_t n = b->size;
    //Vector r = vec_empty(n), r0 = vec_empty(n), p = vec_empty(n), v = vec_empty(n);
    //Vector t = vec_empty(n), s = vec_empty(n), z = vec_empty(n), y = vec_empty(n);
	
	size_t size_with_padding = (n%ALIGNED_ELEM == 0)? 
							n:
							(n/ALIGNED_ELEM+1)*ALIGNED_ELEM;
	alignas(32) double* memory = (double*) xaligned_malloc(8*sizeof(double)*size_with_padding);
	Vector   r={.size=n, .ptr=memory+0*size_with_padding}, 
		    r0={.size=n, .ptr=memory+1*size_with_padding},
		     p={.size=n, .ptr=memory+2*size_with_padding}, 
		     v={.size=n, .ptr=memory+3*size_with_padding}, 
		     t={.size=n, .ptr=memory+4*size_with_padding},
		     s={.size=n, .ptr=memory+5*size_with_padding},
		     z={.size=n, .ptr=memory+6*size_with_padding},
		     y={.size=n, .ptr=memory+7*size_with_padding};
	
    vzero(p.ptr,n);
    vzero(v.ptr,n);
    // r0 = b - A*x   
	apply_both(Band, Other, x->ptr, r.ptr);
	
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i){
		r.ptr[i] = b->ptr[i] - r.ptr[i];
    //memcpy(r0.ptr, r.ptr, sizeof(double)*n);
		r0.ptr[i] = r.ptr[i];
	}
    double rho=1.0, alpha=1.0, omega=1.0;

    double bnrm2 = fmax(1e-30, vnorm2(b->ptr,n));
    double resid = vnorm2(r.ptr,n)/bnrm2;
	
    for(unsigned int it=1; it<=MAXITER; ++it){
        if (CONSOLE_OUTPUT && (it%100 == 0))
			printf("%d: %lf\n", it, resid);
        double rho1 = vdot(r0.ptr, r.ptr, n);
        double beta = (rho1/rho)*(alpha/omega);
        // p = r + beta*(p - omega*v)
		//vaxpy(p.ptr, -omega, v.ptr, n);
		//vxpay(p.ptr, beta, r.ptr,n);
        //#pragma omp parallel for schedule(static)
        //for(uint32_t i=0;i<n;++i) 
		//	p.ptr[i] = r.ptr[i] + beta*(p.ptr[i] - omega*v.ptr[i]);
		
		#pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i){
			p.ptr[i] = p.ptr[i] - omega*v.ptr[i];
			p.ptr[i] = r.ptr[i] + beta*p.ptr[i];
			y.ptr[i] = Minv->ptr[i] * p.ptr[i];
		}
		
        //apply_precond(Minv, p.ptr, y.ptr);
		
		double r0v = 0.0;

		#pragma omp parallel for reduction(+:r0v) schedule(guided)
		for (uint32_t s = 0; s < Band->stripes_len; ++s) {
			uint32_t i1 = Band->stripes[s];
			uint32_t i2 = Band->stripes[s+1];
			
			_mm_prefetch(&x[Band->indices[Band->indptr[i1]]], _MM_HINT_T1);
			
			for (uint32_t i = i1; i < i2; ++i) {
				double sum = 0.0;
				
				// умножение строки из полосы Band
				for (uint32_t jj = Band->indptr[i]; jj < Band->indptr[i+1]; ++jj) {
					sum += Band->data[jj] * y.ptr[Band->indices[jj]];
				}

				// добавляем элементы из Other
				for (uint32_t jj = Other->indptr[i]; jj < Other->indptr[i+1]; ++jj) {
					sum += Other->data[jj] * y.ptr[Other->indices[jj]];
				}

				v.ptr[i] = sum;
				r0v += r0.ptr[i] * sum;
			}
		}
        alpha = rho1 / r0v;

        // s = r - alpha*v
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n; ++i){
			s.ptr[i] = r.ptr[i] - alpha*v.ptr[i];
			z.ptr[i] = Minv->ptr[i]*s.ptr[i];
		

        // Early convergence check
        //double snrm = vnorm2(s.ptr,n)/bnrm2;

        // z = M * s
        //apply_precond(Minv, s.ptr, z.ptr);
		}
        // t = A*z
		double tt = 0.0, ts = 0.0;

		#pragma omp parallel for reduction(+:tt, ts) schedule(guided)
		for (uint32_t s_band = 0; s_band < Band->stripes_len; ++s_band) {
			uint32_t i1 = Band->stripes[s_band];
			uint32_t i2 = Band->stripes[s_band+1];
			
			_mm_prefetch(&x[Band->indices[Band->indptr[i1]]], _MM_HINT_T1);

			for (uint32_t i = i1; i < i2; ++i) {
				double sum = 0.0;

				// умножение строки из полосы Band
				for (uint32_t jj = Band->indptr[i]; jj < Band->indptr[i+1]; ++jj) {
					sum += Band->data[jj] * z.ptr[Band->indices[jj]];
				}

				// добавляем элементы из Other
				for (uint32_t jj = Other->indptr[i]; jj < Other->indptr[i+1]; ++jj) {
					sum += Other->data[jj] * z.ptr[Other->indices[jj]];
				}

				t.ptr[i] = sum;

				// сразу добавляем вклады в редукцию
				tt += sum * sum;
				ts += sum * s.ptr[i];
			}
		}
		
        omega = ts/tt;
		resid = 0;
		
		#pragma omp parallel for reduction(+:resid) schedule(static)
		for(uint32_t i=0;i<n;++i){
			x->ptr[i] += alpha * y.ptr[i];
			x->ptr[i] += omega * z.ptr[i];
        // x = x + alpha*y + omega*z
		//vaxpy(x->ptr, alpha, y.ptr, n);
		//vaxpy(x->ptr, omega, z.ptr, n);
			r.ptr[i] = s.ptr[i] - omega*t.ptr[i];
		

        // r = s - omega*t
        //#pragma omp parallel for schedule(static)
        //for(uint32_t i=0;i<n;++i)
		//	r.ptr[i] = s.ptr[i] - omega*t.ptr[i];
	
			resid += r.ptr[i]*r.ptr[i];
		}
		resid = sqrt(resid)/bnrm2;

        //resid = vnorm2(r.ptr,n)/bnrm2;

        rho = rho1;
    }

    *out_final_res = resid; 
    //vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y);
    xaligned_free(memory);
    return 1; // not fully converged
}

static int bicgstab_simplesolve(const CSRMatrix* A, const Vector* Minv,
                          const Vector* b, Vector* x, unsigned int* iters, double* out_final_res){
	const uint32_t n = b->size;
    //Vector r = vec_empty(n), r0 = vec_empty(n), p = vec_empty(n), v = vec_empty(n);
    //Vector t = vec_empty(n), s = vec_empty(n), z = vec_empty(n), y = vec_empty(n);
	size_t size_with_padding = (n%ALIGNED_ELEM == 0)? 
							n:
							(n/ALIGNED_ELEM+1)*ALIGNED_ELEM;
	alignas(32) double* memory = (double*) xaligned_malloc(8*sizeof(double)*size_with_padding);
	Vector   r={.size=n, .ptr=memory+0*size_with_padding}, 
		    r0={.size=n, .ptr=memory+1*size_with_padding},
		     p={.size=n, .ptr=memory+2*size_with_padding}, 
		     v={.size=n, .ptr=memory+3*size_with_padding}, 
		     t={.size=n, .ptr=memory+4*size_with_padding},
		     s={.size=n, .ptr=memory+5*size_with_padding},
		     z={.size=n, .ptr=memory+6*size_with_padding},
		     y={.size=n, .ptr=memory+7*size_with_padding};
	
    vzero(p.ptr,n);
    vzero(v.ptr,n);
	
    // r0 = b - A*x   
	csr_apply(A, x->ptr, r.ptr);
	
    #pragma omp parallel for schedule(static)
    for(uint32_t i=0;i<n;++i){
		r.ptr[i] = b->ptr[i] - r.ptr[i];
		r0.ptr[i] = r.ptr[i];
	}
    //memcpy(r0.ptr, r.ptr, sizeof(double)*n);

    double rho=1.0, alpha=1.0, omega=1.0;

    double bnrm2 = fmax(1e-30, vnorm2(b->ptr,n));
    double resid = vnorm2(r.ptr,n)/bnrm2;

    for(unsigned int it=1; it<=MAXITER; ++it){
        if (CONSOLE_OUTPUT && (it%100 == 0))
			printf("%d: %lf\n", it, resid);
        double rho1 = vdot(r0.ptr, r.ptr, n);
        double beta = (rho1/rho)*(alpha/omega);
		
        // p = r + beta*(p - omega*v)
		//vaxpy(p.ptr, -omega, v.ptr, n);
		//vxpay(p.ptr, beta, r.ptr,n);
		
		#pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n;++i){
			p.ptr[i] = p.ptr[i] - omega*v.ptr[i];
			p.ptr[i] = r.ptr[i] + beta*p.ptr[i];
			y.ptr[i] = Minv->ptr[i] * p.ptr[i];
		}

        //apply_precond(Minv, p.ptr, y.ptr);
		
		csr_apply(A, y.ptr, v.ptr);
		
        double r0v = vdot(r0.ptr, v.ptr, n);
        alpha = rho1 / r0v;

        // s = r - alpha*v
        #pragma omp parallel for schedule(static)
        for(uint32_t i=0;i<n; ++i) 
			s.ptr[i] = r.ptr[i] - alpha*v.ptr[i];

        // Early convergence check
       // double snrm = vnorm2(s.ptr,n)/bnrm2;

        // z = M * s
        apply_precond(Minv, s.ptr, z.ptr);

        // t = A*z
		csr_apply(A, z.ptr, t.ptr);
		
		
        double tt = 0, ts = 0;
		#pragma omp parallel for reduction(+:tt, ts) schedule(static)
		for(uint32_t i=0;i<n;++i){
			tt += t.ptr[i]*t.ptr[i];
			ts += t.ptr[i]*s.ptr[i];
			
			//double tt = vdot(t.ptr,t.ptr,n);
			//double ts = vdot(t.ptr,s.ptr,n);
		}
		
        omega = ts/tt;
		resid = 0;
		
		#pragma omp parallel for schedule(static)
		for(uint32_t i=0;i<n;++i){
			x->ptr[i] += alpha * y.ptr[i];
			x->ptr[i] += omega * z.ptr[i];
        // x = x + alpha*y + omega*z
		//vaxpy(x->ptr, alpha, y.ptr, n);
		//vaxpy(x->ptr, omega, z.ptr, n);
			r.ptr[i] = s.ptr[i] - omega*t.ptr[i];
		

        // r = s - omega*t
        //#pragma omp parallel for schedule(static)
        //for(uint32_t i=0;i<n;++i)
		//	r.ptr[i] = s.ptr[i] - omega*t.ptr[i];
	
			resid += r.ptr[i]*r.ptr[i];
		}
		resid = sqrt(resid)/bnrm2;

        //resid = vnorm2(r.ptr,n)/bnrm2;

        rho = rho1;
    }

    *out_final_res = resid; 
    //vec_free(&r); vec_free(&r0); vec_free(&p); vec_free(&v); vec_free(&t); vec_free(&s); vec_free(&z); vec_free(&y);
    xaligned_free(memory);
    return 1; // not fully converged
}

// ===================== Main =====================
int main(int argc, char** argv){
    const char* MATFILE = "pwtk.mat";
    const char* STRUCT  = "Problem"; 
    const char* FIELD   = "A";      
    const char* BFILE   = "b.bin";
    const char* XFILE   = "x.bin";
    const char* MINV_FILE = "diag.bin";
	char COMPARISON = 0;
	if (argc >= 2){
		if (strcmp (argv[1],"NODEBUG") == 0)
			CONSOLE_OUTPUT = 0;
		else
			puts("DEBUG");
		
	}
	if (argc >= 3)
		THREADS_NUM = (unsigned)strtoul(argv[2], NULL, 10);
	if (argc >= 4)
		MAXITER = (unsigned)strtoul(argv[3], NULL, 10);
	if (argc >=5){
		k = (double) strtod(argv[4], NULL);
	}
	if (argc >= 6){
		if (strcmp (argv[5],"PRECONDITIONER") == 0)
			COMPARISON = 1;
		else if (strcmp (argv[5],"MATRIX") == 0)
			COMPARISON = 2;
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

    StripeMatrix A_band={}, A_other={};
	CSRMatrix A;
    
	if (COMPARISON == 2)
		csc_to_csr_single(input, N, &A);
	else
		csc_to_csr_and_split(input, N, &A_band, &A_other);
    Vector Minv;
	if (COMPARISON == 1){
		Minv = vec_empty(N);
		for (unsigned int i = 0; i < N; ++i)
			Minv.ptr[i] = 1;
		if (CONSOLE_OUTPUT)
			printf("DEBUG: using identity preconditioner (should reduce to unpreconditioned BiCGStab)\n");
	}
	else{
		// build or load diagonal preconditioner (as Vector)
		if(load_vector_binary(MINV_FILE, &Minv)){
		   // loaded
		} else {
			printf("Preconditioner file not found — building...\n");
			spai0_build_diagonal(input, N, &Minv);
			printf("Preconditioner built. saving...\n");
			if(!save_vector_binary(MINV_FILE, &Minv)) fprintf(stderr, "Warning: failed to save preconditioner to %s\n", MINV_FILE);
			check_AM_is_I(&A_band, &A_other, &Minv, N);
		}
	}
    // input matrix memory no longer needed
    Mat_VarFree(field);
    Vector b = make_or_load_b(BFILE, N);
    Vector x = vec_empty(N); 
	vzero(x.ptr, N);
	int status = 0;
	if (COMPARISON == 2){
		compute_print_true_residual_single(&A, &b, &x);
		double t0 = wall_seconds();
		unsigned int iters=0; 
		double final_res=0.0;
		
		status = bicgstab_simplesolve(&A, &Minv, &b, &x, &iters, &final_res);
		double t1 = wall_seconds();
		if (CONSOLE_OUTPUT)
			printf("Iterations: %u\n", iters);
		else
			printf("Threads: %u\n", THREADS_NUM);
		
		printf("Final relative residual: %.6e\n", final_res);
		printf("Elapsed (s): %.6f\n", (t1-t0));

		compute_print_true_residual_single(&A, &b, &x);
	}
	else{
		compute_print_true_residual(&A_band, &A_other, &b, &x);
		double t0 = wall_seconds();
		unsigned int iters=0; 
		double final_res=0.0;
		
		status = bicgstab_solve(&A_band, &A_other, &Minv, &b, &x, &iters, &final_res);
		double t1 = wall_seconds();
		if (CONSOLE_OUTPUT)
			printf("Iterations: %u\n", iters);
		else
			printf("Threads: %u\n", THREADS_NUM);
		
		printf("Final relative residual: %.6e\n", final_res);
		printf("Elapsed (s): %.6f\n", (t1-t0));

		compute_print_true_residual(&A_band, &A_other, &b, &x);
	}

    write_binary_vector(XFILE, &x);

    vec_free(&x); vec_free(&b);
	if (COMPARISON == 2)
		csr_free(&A);
	else{
		stripe_free(&A_band); stripe_free(&A_other);
	}
	vec_free(&Minv);
    return 0;
}
