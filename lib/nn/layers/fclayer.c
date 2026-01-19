#include "fclayer.h"

#include <stdlib.h>
#include <math.h>

Matrix *he_weights(int input_size, int output_size)
{
    return random_normal_matrix(input_size, output_size, sqrt(2.0 / input_size));
}

FCLayer *init_fc_layer(int input_size, int output_size)
{
    FCLayer *l = malloc(sizeof(FCLayer));

    if (!l)
        return NULL;

    l->type = LAYER_FULLYCONNECTED;

    l->weights = he_weights(input_size, output_size);
    l->bias = he_weights(1, output_size);

    l->input_size = input_size;
    l->output_size = output_size;

    l->in = NULL;
    l->out = NULL;

    return l;
}

void forward_fc_layer(FCLayer *l, Matrix *x)
{
    free_matrix(l->in);
    free_matrix(l->out);

    l->in = copy_matrix(x);

    Matrix *mulled = mul_matrix(x, l->weights);
    l->out = add_matrices(mulled, l->bias);

    free_matrix(mulled);
}

Matrix *backward_fc_layer(FCLayer *l, Matrix *output_error, float lr)
{
    Matrix *weights_tr = trans_matrix(l->weights);
    Matrix *in_error = mul_matrix(output_error, weights_tr);
    free(weights_tr);

    Matrix *input_tr = trans_matrix(l->in);
    Matrix *weights_error = mul_matrix(input_tr, output_error);
    free(input_tr);

    mul_c_matrix(weights_error, lr);
    mul_c_matrix(output_error, lr);

    Matrix *new_weights = sub_matrices(l->weights, weights_error);
    Matrix *new_bias = sub_matrices(l->weights, weights_error);

    free(l->weights);
    free(l->bias);

    l->weights = new_weights;
    l->bias = new_bias;

    free(output_error);
    free(weights_error);

    return in_error;
}

void free_fc_layer(FCLayer *l)
{
    free_matrix(l->weights);
    free_matrix(l->bias);

    free_matrix(l->in);
    free_matrix(l->out);
}