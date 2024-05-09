/*This Program has the Parallel implementation of SVM Algorithm using MPI*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>


void read_csv(char *filename, int num_data_points, int num_features, double data[num_data_points][num_features+1]){

    /*Function that reads the csv data and poppulates an array*/

    FILE *file = fopen(filename, "r");
    if(file == NULL){
        printf("Error: File not found\n");
        exit(1);
    }

    int i = 0;
    char line[1024];
    while(fgets(line, sizeof(line), file)){
        // printf("%s", line);
        if(i>0){
            char *token;
            int count = 0;
            token = strtok(line, ",");
            while (token != NULL) {
                if (count >= 1 && count <= num_features) {
                    data[i-1][count-1] = atof(token);
                }
                if(count==num_features+1){
                    char *end = strtok(token, "\n");
                    data[i-1][count-1] = atof(end);
                }
                /*When strtok is called with NULL as the first argument, it continues tokenizing the same string from where it left off in the previous call.*/
                token = strtok(NULL, ",");
                count++;
            }
            // printf("Data point %d is %f %f %f\n", i, data[i-1][0], data[i-1][1], data[i-1][2]);

        }
        i++;
    }
}



int main(int argc, char **argv){

    printf("Running the Parallel MPI version of SVM code\n");
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    double b = 0.0;
    double alpha = 0.001;
    int num_iterations = 10;
    double lamda = 0.01;
    int num_points = 200;
    int num_features = 2;
    double data[num_points][num_features+1];
    double w[num_features];
    double global_w[num_features];
    double global_b = 0.0;

    if(rank == 0){
        // data = (double **)malloc(num_points*sizeof(double *));
        // for(int i=0; i<num_points; i++){
        //     data[i] = (double *)malloc((num_features+1)*sizeof(double));
        // }
        char filename[100];
        sprintf(filename, "../Data/Two_class/data.csv");


        //populating the array with the data
        read_csv(filename, num_points, num_features, data);
        printf("Data read successfully\n");
        int i=1;
        printf("Data point %d is %f %f %f\n", i, data[i-1][0], data[i-1][1], data[i-1][2]);
    }


    //Master node reads the data and distributes it to the worker nodes

    int n_num_points = num_points/size;
    double local_data[n_num_points][num_features+1];

    MPI_Scatter(data, n_num_points*(num_features+1), MPI_DOUBLE, local_data, n_num_points*(num_features+1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printf("Data Scattered\n");
    printf("Sanity check: Data point %d in rank %d is %f %f %f\n", 1, rank, local_data[0][0], local_data[0][1], local_data[0][2]);

    //initializing the weights and bias
    for(int i=0; i<num_features; i++){
        w[i] = 0;
        global_w[i] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // for(int i=0; i<num_features;i++){
    //     printf("w[%d] = %f, rank = %d\n", i, w[i],rank);
    // }
    for(int i=0; i<num_iterations; i++){
        
        for(int j=0; j<n_num_points; j++){
            double y = local_data[j][num_features];
            double sum = 0;
            for(int k=0; k<num_features; k++){
                // printf("local_data[%d][%d] is %f, w[%d] = %f\n", j, k, local_data[j][k], k ,w[k]);
                sum += w[k]*local_data[j][k];
            }

            double z = y*(sum - b);
            // printf("sum is %f, b is %f, y is %f, z is %f, rank = %d\n",sum, b,y,z, rank);
            if(z < 1){
                for(int k=0; k<num_features; k++){
                    w[k] = w[k] - alpha*(2.0*lamda*w[k] - y*local_data[j][k]);
                }
                b = b - alpha*y;
            }
            else{
                for(int k=0; k<num_features; k++){
                    w[k] = w[k] - alpha*(2.0*lamda*w[k]);
                }
            }
        }
        printf("b is %f, rank = %d\n", b, rank);
        MPI_Reduce(w, global_w, num_features, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&b, &global_b, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Bcast(global_w, num_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for(int k=0; k<num_features; k++){
            w[k] = global_w[k];
        }
        b = global_b;
        MPI_Barrier(MPI_COMM_WORLD);
        printf("after one complete iter, b is %f, rank = %d\n", b, rank);

    }

    if(rank==0){
                printf("Saving the data to file\n");
            FILE *file = fopen("../model/Two_class/model_MPI.csv", "w");
            if(file == NULL){
                printf("Error: File not found\n");
                exit(1);
            }
            fprintf(file,"w,b\n");
            for(int i=0; i<num_features; i++){
                fprintf(file, "%f,%f\n", w[i], b);
            }
            fclose(file);
    }

    MPI_Finalize();

    return 0;
}