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
    sprintf(filename, "../Data/IRIS/Iris.csv");

    int num_points = 150;
    int num_features = 4;
    double data[num_points][num_features];

    //populating the array with the data
    read_csv(filename, num_points, num_features, data);

    

    return 0;
}