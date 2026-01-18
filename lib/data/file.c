#include "file.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "dataset.h"
#include "../math/matrix.h"

enum
{
    max_line_length = 100,
};

const char *delims = ";,";

Dataset *from_file_dataset(char *file_name)
{
    FILE *fptr;

    fptr = fopen(file_name, "r");

    if (!fptr)
        return NULL;

    char line[max_line_length];

    fgets(line, max_line_length, fptr);

    char *tok = strtok(line, delims);
    int input_size = atoi(tok);

    tok = strtok(NULL, delims);
    int output_size = atoi(tok);

    Dataset *dataset = init_dataset(input_size, output_size);

    while (fgets(line, max_line_length, fptr))
    {
        float *x = malloc(sizeof(float) * input_size);
        if (!x)
            return NULL;

        tok = strtok(line, delims);
        for (int i = 0; i < input_size; ++i)
        {
            x[i] = atof(tok);
            tok = strtok(NULL, delims);
        }

        Matrix *x_mat = to_matrix(x, input_size, input_size, 1);

        float *y = malloc(sizeof(float) * output_size);
        if (!y)
            return NULL;

        for (int i = 0; i < output_size; ++i)
        {
            y[i] = atof(tok);
            tok = strtok(NULL, delims);
        }

        Matrix *y_mat = to_matrix(y, output_size, output_size, 1);

        add_to_dataset(dataset, x_mat, y_mat);
    }

    fclose(fptr);

    return dataset;
}