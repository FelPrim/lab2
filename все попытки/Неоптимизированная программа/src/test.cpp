#include <cstdio>
#include <cstring>
#include <cassert>

#include "useful_math.hpp"
#include "vector.hpp"
#include "sparsematrix.hpp"
#include "preconditioner.hpp"
#include "bicg.hpp"

int run_useful_math(){
    puts("Yeeee");
    return 0;    
}

int run_vector(){
    Vector a;
    Vector b(5);
    numeric number = 23;
    for (unsigned int i = 0; i < b.size(); ++i){
        b[i] = number;
        number *= -1.2;
    }
    Vector c = b;
    b[2] = 3;
    assert(c[2] != 3);
    c[2] = 3;
    a = b*numeric;
    c *= numeric;
    for (unsigned int i = 0; i < b.size(); ++i)
        assert(a[i] == c[i]);
    
    
    return 0;
}

int run_sparsematrix(){
    return 0;
}

int run_preconditioner(){
    return 0;
}

int run_bicg(){
    retunr 0;
}

static inline bool str_is__(const char *str, const char *name){
    return strcmp(str, name) == 0;
}

int main(int argc, const char *argv[]){

    bool RUN_USEFUL_MATH = 1;
    bool RUN_VECTOR = 1;
    bool RUN_SPARSEMATRIX = 1;
    bool RUN_PRECONDITIONER = 1;
    bool RUN_BICG = 1;

    if (argc > 1){

        RUN_USEFUL_MATH = 0;
        RUN_VECTOR = 0;
        RUN_SPARSEMATRIX = 0;
        RUN_PRECONDITIONER = 0;
        RUN_BICG = 0;

        for (int i = 1; i < argc; ++i){
            RUN_USEFUL_MATH |= str_is__(argv[i], "useful_math");
            RUN_VECTOR |= str_is__(argv[i], "vector");
            RUN_SPARSEMATRIX |= str_is__(argv[i], "sparsematrix");
            RUN_USEFUL_MATH |= str_is__(argv[i], "preconditioner");
            RUN_BICG |= str_is__(argv[i], "bicg");
        }
    }

    if (RUN_USEFUL_MATH) run_useful_math(); 
    if (RUN_VECTOR) run_vector();
    if (RUN_SPARSEMATRIX) run_sparsematrix();
    if (RUN_PRECONDITIONER) run_preconditioner();
    if (RUN_BICG) run_bicg();

    return 0;
}
