#include<stdio.h>
#include <string.h>

int main() {
   char string[50] = "1/n";
   // Extract the first token
   char * token = strtok(string, "/n");
   printf( " %s\n", token ); //printing the token
   return 0;
}
