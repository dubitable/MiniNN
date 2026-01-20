#include "fclayer.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../../math/matrix.h"
#include "layertype.h"

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
    l->bias = zeros_matrix(1, output_size);

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
    free_matrix(weights_tr);

    Matrix *input_tr = trans_matrix(l->in);
    Matrix *weights_error = mul_matrix(input_tr, output_error);
    free_matrix(input_tr);

    mul_c_matrix(weights_error, lr);

    Matrix *new_weights = sub_matrices(l->weights, weights_error);

    Matrix *bias_error = copy_matrix(output_error);
    mul_c_matrix(bias_error, lr);
    Matrix *new_bias = sub_matrices(l->bias, bias_error);

    free_matrix(l->weights);
    free_matrix(l->bias);

    l->weights = new_weights;
    l->bias = new_bias;

    free_matrix(weights_error);
    free_matrix(bias_error);

    return in_error;
}

void print_fc_layer()
{
    printf("FC");
}

int num_params_fc_layer(FCLayer *l)
{
    return (l->weights->dims.h * l->weights->dims.w) + (l->bias->dims.h * l->bias->dims.w);
}

void free_fc_layer(FCLayer *l)
{
    free_matrix(l->weights);
    free_matrix(l->bias);

    free_matrix(l->in);
    free_matrix(l->out);
}