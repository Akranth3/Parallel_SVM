#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
// #include <omp.h> // Include OpenMP library

#ifdef _OPENACC
#include <openacc.h>
#endif

int main(void)
{
    
    int N = 100;
    double *a = (double *)malloc(N * sizeof(double));
    for(int i=0; i<N; i++){
        a[i] = i;
    }

    double sums = 0;
    double *b = (double *)malloc(N * sizeof(double));

    #pragma acc data copyout(b[0:N])
    {
        #pragma acc parallel loop gang
            for(int i=0; i<N; i++){
                sums = 0;
                #pragma acc loop vector
                for(int j=0; j<N; j++){
                    sums += a[i]*a[j];
                }
                b[i] = sums;
            }
    }
    
    for(int i=0; i<N; i++){
        printf("%f\n", b[i]);
    }

    return 0;
}