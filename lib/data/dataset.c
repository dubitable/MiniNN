#include "dataset.h"

#include <stdlib.h>

#include "../math/matrix.h"
#include "../math/random.h"

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

DatasetSplit *train_test_val(Dataset *dataset, float train_prop, float test_prop, int is_random)
{
    int train_count = dataset->count * train_prop;
    int test_count = dataset->count * test_prop;

    int *order = generate_order(dataset->count);

    if (is_random)
        shuffle(order, dataset->count);

    int i = 0;

    DatasetSplit *s = malloc(sizeof(DatasetSplit));
    s->count = dataset->count;

    Dataset *train = init_dataset(dataset->input_size, dataset->output_size);
    for (int j = i; j < train_count; ++j)
    {
        add_to_dataset(train, dataset->x[order[i]], dataset->y[order[i]]);
        i++;
    }

    s->train = train;

    Dataset *test = init_dataset(dataset->input_size, dataset->output_size);
    for (int j = i; j < test_count + train_count; ++j)
    {
        add_to_dataset(test, dataset->x[order[i]], dataset->y[order[i]]);
        i++;
    }

    s->test = test;

    Dataset *val = init_dataset(dataset->input_size, dataset->output_size);
    for (int j = i; j < dataset->count; ++j)
    {
        add_to_dataset(val, dataset->x[order[i]], dataset->y[order[i]]);
        i++;
    }

    s->val = val;

    free(dataset);
    free(order);

    return s;
}

void free_datasetsplit(DatasetSplit *split)
{
    free_dataset(split->train);
    free_dataset(split->test);
    free_dataset(split->val);
    free(split);
}