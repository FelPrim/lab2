#include <stdio.h>
#include <string.h>
#include "math_stuff.h"



int run_a(){
    return 0;    
}


static inline bool str_is__(const char *str, const char *name){
    return strcmp(str, name) == 0;
}

int main(int argc, const char *argv[]){
    bool RUN_A = 1;

    if (argc > 1){
        for (int i = 1; i < argc; ++i){
            RUN_A = str_is__(argv[i], "");
        }
    }

    if (RUN_A) run_a();

    return 0;
}
