#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../math/matrix.h"

Matrix *he_weights(int size, int input_size)
{
    return random_normal_matrix(size, input_size + 1, sqrt(2.0 / input_size));
}

Layer *init_layer(int size, int input_size)
{
    Layer *l = malloc(sizeof(Layer));
    Matrix *m = he_weights(size, input_size);

    l->weights = m;
    l->size = size;
    l->input_size = input_size;

    return l;
}

Matrix *forward_layer(Layer *l, Matrix *x)
{
    add_ones_row_matrix(x);
    Matrix *a = mul_matrix(l->weights, x);
    relu_matrix(a);
    return a;
}

void print_layer(Layer *l)
{
    printf("(%d, %d)\n", l->size, l->input_size + 1);
}

void free_layer(Layer *l)
{
    free_matrix(l->weights);
    free(l);
}