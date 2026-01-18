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

    if (!l)
        return NULL;

    Matrix *m = he_weights(size, input_size);

    l->weights = m;
    l->size = size;
    l->input_size = input_size;

    return l;
}

void forward_layer(Layer *l, Matrix *x)
{
    Matrix *x_b = copy_matrix(x);
    add_ones_row_matrix(x_b);

    Matrix *a = mul_matrix(l->weights, x_b);
    relu_matrix(a);

    free_matrix(x_b);

    l->out = a;
}

Matrix *backward_layer(Layer *layer, Matrix *dL_dout, Matrix *input)
{
    Matrix *x_b = copy_matrix(input);
    add_ones_row_matrix(x_b);

    Matrix *relu_mask = copy_matrix(layer->out);
    reluder_matrix(relu_mask);

    Matrix *dL_da = mul_matrices(dL_dout, relu_mask);
    Matrix *x_b_t = trans_matrix(x_b);

    free_matrix(layer->grad_weights);
    free_matrix(x_b);

    layer->grad_weights = mul_matrix(dL_dout, x_b_t);

    Matrix *weights_t = trans_matrix(layer->weights);
    Matrix *dL_dx = mul_matrix(weights_t, dL_da);

    remove_last_matrix(dL_dx);

    free_matrix(x_b_t);
    free_matrix(weights_t);
    free_matrix(relu_mask);

    return dL_dx;
}

void update_layer(Layer *layer, float lr)
{
    Matrix *grad_weights = copy_matrix(layer->grad_weights);
    mul_c_matrix(grad_weights, lr);

    Matrix *new_weights = sub_matrices(layer->weights, grad_weights);

    free_matrix(layer->weights);
    free_matrix(grad_weights);

    layer->weights = new_weights;
}

void print_layer(Layer *l)
{
    printf("(%d, %d)\n", l->size, l->input_size + 1);
}

void free_layer(Layer *l)
{
    free_matrix(l->weights);

    if (l->out)
    {
        free_matrix(l->out);
    }

    if (l->grad_weights)
    {
        free_matrix(l->grad_weights);
    }

    free(l);
}