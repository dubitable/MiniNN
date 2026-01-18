#ifndef DATASET_H_INCLUDED
#define DATASET_H_INCLUDED

#include "dataset.h"
#include "../math/matrix.h"

typedef struct
{
    Matrix **x;
    Matrix **y;

    int count;

    int input_size;
    int output_size;
} Dataset;

typedef struct
{
    Matrix *x;
    Matrix *y;
} DataSample;

Dataset *init_dataset(int, int);
DataSample sample_dataset(Dataset *);

void add_to_dataset(Dataset *, Matrix *, Matrix *);

void free_dataset(Dataset *);

#endif