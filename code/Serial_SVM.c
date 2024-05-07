/*This Program has the serial implementation of SVM Algorithm*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>



void read_csv(char *filename, int num_data_points, int num_features, double data[num_data_points][num_features]){

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
                if (count >= 1 && count <= 4) {
                    data[i-1][count-1] = atof(token);
                }
                /*When strtok is called with NULL as the first argument, it continues tokenizing the same string from where it left off in the previous call.*/
                token = strtok(NULL, ",");
                count++;
            }
        }
        i++;
    }
}



int main(){
    printf("Running the serial SVM code\n");
    
    char filename[100];
    sprintf(filename, "../Data/Two_class/data.csv");

    int num_points = 200;
    int num_features = 2; 
    double data[num_points][num_features+1]; // 2 features and 1 label

    //populating the array with the data
    read_csv(filename, num_points, num_features, data);
    printf("Data read successfully\n");
    printf("the first data point is %f %f %f\n", data[0][0], data[0][1], data[0][2]);
    
    double w[num_features-1];
    double b = 0.0;
    double alpha = 0.001;
    int num_iterations = 1000;
    double lamda = 0.01;

    //initializing the weights and bias
    for(int i=0; i<num_features-1; i++){
        w[i] = 0;
    }

    for(int i=0; i<num_iterations; i++){
        for(int j=0; j<num_points; j++){
            double y = data[j][num_features];
            double sum = 0;
            for(int k=0; k<num_features-1; k++){
                sum += w[k]*data[j][k];
            }
            double z = y*(sum - b);
            if(z < 1){
                for(int k=0; k<num_features-1; k++){
                    w[k] = w[k] - alpha*(2*lamda*w[k] - y*data[j][k]);
                }
                b = b - alpha*y;
            }
            else{
                for(int k=0; k<num_features-1; k++){
                    w[k] = w[k] - alpha*(2*lamda*w[k]);
                }
            }
        }

    }
    FILE *file = fopen("../model/Two_class/model.csv", "w");
    if(file == NULL){
        printf("Error: File not found\n");
        exit(1);
    }
    fprintf(file,"w,b\n");
    for(int i=0; i<num_features-1; i++){
        fprintf(file, "%f,%f\n", w[i], b);
    }
    fclose(file);




    return 0;
}