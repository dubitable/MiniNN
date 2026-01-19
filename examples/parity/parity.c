#include <stdio.h>
#include <stdlib.h>

const int count_default = 1000;

int main(int argc, char **argv)
{
    FILE *fptr;

    fptr = fopen("parity.mini", "w");

    int count = count_default;

    if (argc == 2)
    {
        count = atoi(argv[1]);
    }

    fprintf(fptr, "1,2\n");

    for (int i = 0; i < count; ++i)
    {
        fprintf(fptr, "%d;", i);
        fprintf(fptr, "%d,%d\n", i % 2 == 0, i % 2 == 1);
    }

    return 0;
}