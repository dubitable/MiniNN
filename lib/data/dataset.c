#include "dataset.h"

#include <stdlib.h>

#include "../math/matrix.h"

Dataset *init_dataset(int input_size, int output_size)
{
    Dataset *dataset = malloc(sizeof(Dataset));

    if (!dataset)
        return NULL;

    dataset->count = 0;
    dataset->input_size = input_size;
    dataset->output_size = output_size;

    return dataset;
}

DataSample sample_dataset(Dataset *dataset)
{
    int index = rand() % dataset->count;

    return (DataSample){
        .x = dataset->x[index],
        .y = dataset->y[index]};
}

void add_to_dataset(Dataset *dataset, Matrix *x, Matrix *y)
{
    dataset->x = realloc(dataset->x, sizeof(Matrix *) * (dataset->count + 1));
    dataset->y = realloc(dataset->y, sizeof(Matrix *) * (dataset->count + 1));

    dataset->x[dataset->count] = x;
    dataset->y[dataset->count] = y;

    dataset->count++;
}

void free_dataset(Dataset *dataset)
{
    for (int i = 0; i < dataset->count; ++i)
    {
        free_matrix(dataset->x[i]);
        free_matrix(dataset->y[i]);
    }

    free(dataset->x);
    free(dataset->y);

    free(dataset);
}