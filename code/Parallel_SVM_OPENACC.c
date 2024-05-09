/*This Program has the serial implementation of SVM Algorithm*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <openacc.h>



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



int main(){
    printf("Running the Parallel-OPENACC SVM code\n");
    
    char filename[100];
    sprintf(filename, "../Data/Two_class/data.csv");

    int num_points = 200;
    int num_features = 2; 
    double data[num_points][num_features+1]; // 2 features and 1 label

    //populating the array with the data
    read_csv(filename, num_points, num_features, data);
    printf("Data read successfully\n");
    int i=1;
    printf("(Sanity check) Data point %d is %f %f %f\n", i, data[i-1][0], data[i-1][1], data[i-1][2]);
    

    //initializing the weights and bias
    double w[num_features];
    double b = 0.0;
    double alpha = 0.001;
    int num_iterations = 1000;
    double lamda = 0.01;

    clock_t start_time = clock();     //timing the code


    //training the model
    # pragma acc data  copyin(data[0:num_points][0:num_features+1]) copyout(w[:num_features], b) 
    {
    
    # pragma acc parallel loop 
    for(int i=0; i<num_features-1; i++){
        w[i] = 0;
    }
    printf("Weights initialized\n");

    for(int i=0; i<num_iterations; i++){
        double *z = (double *)malloc(num_points*sizeof(double));
        double sum, y;
        //present(w[:num_features], data[:num_points][:num_features+1], z[:num_points])
        # pragma acc parallel loop gang 
        for(int j=0; j<num_points; j++){
            sum = 0;

            #pragma acc loop vector
            for(int k=0; k<num_features; k++){
                sum += w[k]*data[j][k];
            }

            y = data[j][num_features];
            z[j] = y*(sum - b);
        }
        
        //present(data[:num_points][:num_features+1], w[:num_features], z[:num_points])
        # pragma acc serial 
        for(int j=0; j<num_points; j++){
            y = data[j][num_features];
            if(z[j] < 1){
                for(int k=0; k<num_features; k++)
                    w[k] = w[k] - alpha*(2.0*lamda*w[k] - y*data[j][k]);
                b = b - alpha*y;
                }
            else{
                for(int k=0; k<num_features; k++)
                    w[k] = w[k] - alpha*(2.0*lamda*w[k]);
                }
        }

    }

    }

    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", execution_time);

    FILE *file = fopen("../model/Two_class/model_openacc.csv", "w");
    if(file == NULL){
        printf("Error: File not found\n");
        exit(1);
    }
    fprintf(file,"w,b\n");
    for(int i=0; i<num_features; i++){
        fprintf(file, "%f,%f\n", w[i], b);
    }
    fclose(file);




    return 0;
}