#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    FILE *fptr;

    fptr = fopen("xor.mini", "w");

    fprintf(fptr, "2,1\n");

    for (int i = 0; i < 50; ++i)
    {
        fprintf(fptr, "0,0;1, 0\n");
        fprintf(fptr, "0,1;0, 1\n");
        fprintf(fptr, "1,0;0, 1\n");
        fprintf(fptr, "1,1;1, 0\n");
    }

    return 0;
}