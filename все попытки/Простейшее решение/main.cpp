#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <matio.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

int main(){
    mat_t *matfp = Mat_Open("pwtk.mat", MAT_ACC_RDONLY);
    if (!matfp){
        std::cout << 1 << std::endl;
        return 1;
    }
    matvar_t *sparse_var = Mat_VarRead(matfp, "Problem");
    Mat_Close(matfp);
    if (!matfp){
        std::cout << 2 << std::endl;
        return 2;
    }
    matvar_t *sparse_field = Mat_VarGetStructFieldByName(sparse_var, "A", 0);
    if (!sparse_field){
        std::cout << 3 << std::endl;
        Mat_VarFree(sparse_var);
        return 3;
    }
    if (!sparse_field->data){
        std::cout << 4 << std::endl;
        Mat_VarFree(sparse_field);
        Mat_VarFree(sparse_var);
        return 4;
    }
    mat_sparse_t *mat_rix = static_cast<mat_sparse_t*>(sparse_field->data);
    if (!mat_rix){
        std::cout << 5 << std::endl;
        Mat_VarFree(sparse_field);
        Mat_VarFree(sparse_var);
        return 5;
    }
    unsigned int nrows = (sparse_var->rank >= 1)? sparse_var->dims[0] : 0;
    unsigned int ncols = (sparse_var->rank >= 2)? sparse_var->dims[1] : 0;
    if (nrows == 0 || ncols == 0){
        std::cout << 40 << std::endl;
        return 40;
    }
    unsigned int nnz = mat_rix->nzmax;
    if (nnz == 0){
        std::cout << 6 << std::endl;
        Mat_VarFree(sparse_field);
        Mat_VarFree(sparse_var);
        return 6;
    }
    unsigned int *ir = mat_rix->ir;
    unsigned int *jc = mat_rix->jc;
    if (!ir || !jc){
        std::cout << 7 << std::endl;
        Mat_VarFree(sparse_field);
        Mat_VarFree(sparse_var);
        return 7;
    }
    std::vector<double> values;
    values.reserve(nnz);
    if (sparse_field->data_type == MAT_T_DOUBLE){
        double *dptr = static_cast<double*>(mat_rix->data);
        if (!dptr){
            std::cout << 8 << std::endl;
            Mat_VarFree(sparse_field);
            Mat_VarFree(sparse_var);
            return 8;
        }
        values.assign(dptr, dptr+nnz);
    }
    else{
        std::cout << 9 << std::endl;
        std::cout << sparse_var->data_type << std::endl;
        Mat_VarFree(sparse_field);
        Mat_VarFree(sparse_var);
        return 9;
    }
    Mat_VarFree(sparse_var);

    using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    SpMat A((Eigen::Index)nrows, (Eigen::Index)ncols);
    try {
        A.reserve(nnz);
    } catch (const std::exception& e) {
        std::cout << "Something happened" << std::endl;
        std::cerr << e.what() << std::endl;
        return 10;
    }
    std::cout << "Nothing happened" << std::endl;
    for (unsigned int col = 0; col < ncols; ++col){
        unsigned int start = jc[col];
        unsigned int end = jc[col+1];
        if (end < start){
            std::cout << col << ", " << start << "," << end << "," << ncols << "," << nrows << "\n";
            std::cout << 11 << std::endl;
            return 11;
        }
        for (unsigned int idx = start; idx < end; ++idx){
            unsigned int row = ir[idx];
            if (row >= nrows){
                std::cout << 12 << std::endl;
                return 12;
            }
            double v = values[idx];
            A.insert((Eigen::Index) row, (Eigen::Index) col) = v;
        }
    }
    
    A.makeCompressed();
    std::cout << "Matrix loaded" << std::endl;

    Eigen::Index N = (Eigen::Index) ncols;
    std::vector<double> bvec_storage(N);
    if (fs::exists("rhs.bin")){
        std::ifstream ifs("rhs.bin", std::ios::binary);
        if (!ifs){
            std::cout << 13 << std::endl;
            return 13;
        }
        ifs.seekg(0, std::ios::end);
        auto fsize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        if (fsize != static_cast<std::streamoff>(N * sizeof(double))){
            std::cout << 16 << std::endl;
            return 16;
        }
        ifs.read(reinterpret_cast<char*>(bvec_storage.data()), N * sizeof(double));
        ifs.close();
        std::cout << "Vector loaded" << std::endl;
    } else {
        std::mt19937_64 rng(123456789ULL);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (Eigen::Index i = 0; i < N; ++i)
            bvec_storage[i] = dist(rng);
        std::ofstream ofs("rhs.bin", std::ios::binary);
        if (!ofs){
            std::cout << 17 << std::endl;
            return 17;
        }
        ofs.write(reinterpret_cast<const char*>(bvec_storage.data()), N*sizeof(double));
        ofs.close();
        std::cout << "Vector generated" << std::endl;
    }

    Eigen::Map<Eigen::VectorXd> bvec(bvec_storage.data(), N);

    Eigen::VectorXd x(N);
    Eigen::BiCGSTAB<SpMat> solver;
    solver.setTolerance(1e-8);
    solver.setMaxIterations(10000);
    std::cout << "Computing precondioner\n";
    solver.compute(A);
    if (solver.info() != Eigen::Success){
        std::cerr << "FAILED: solver.compute" << std::endl;
        return 18;
    }
    std::cout << "Solving with BiCGStab...\n";
    x = solver.solve(bvec);

    if (solver.info() != Eigen::Success){
        std::cerr << solver.info() << std::endl;
    }
    std::cout << "Iterations: " << solver.iterations() << ", estimated error = " << solver.error() << "\n";

    std::ofstream ofs("x.bin", std::ios::binary);
    if (!ofs){
        std::cout << 19<< std::endl;
        return 19;
    }
    ofs.write(reinterpret_cast<const char*>(x.data()), N * sizeof(double));
    ofs.close();


    return 0;
}
