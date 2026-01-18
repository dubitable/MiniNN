#include <stdio.h>

int main()
{
    FILE *fptr;

    fptr = fopen("parity.mini", "w");

    int count = 100;

    fprintf(fptr, "1,2\n");

    for (int i = 0; i < count; ++i)
    {
        fprintf(fptr, "%d;", i);
        fprintf(fptr, "%d,%d\n", i % 2 == 0, i % 2 == 1);
    }

    return 0;
}