#include "alayer.h"

#include <stdlib.h>
#include <stdio.h>

#include "../../metrics/activation.h"
#include "../../math/matrix.h"
#include "layertype.h"

ALayer *init_a_layer(int input_size, int output_size, Activation a)
{
    ALayer *l = malloc(sizeof(ALayer));

    if (!l)
        return NULL;

    l->type = LAYER_ACTIVATION;

    ActivationFns a_fns = use_activation(a);

    l->activation_type = a;
    l->activation = a_fns;

    l->input_size = input_size;
    l->output_size = output_size;

    l->in = NULL;
    l->out = NULL;

    return l;
}

void forward_a_layer(ALayer *l, Matrix *x)
{
    free_matrix(l->in);
    free_matrix(l->out);

    l->in = copy_matrix(x);
    l->out = copy_matrix(x);

    l->activation.a(l->out);
}

Matrix *backward_a_layer(ALayer *l, Matrix *output_error)
{
    Matrix *aprime = copy_matrix(l->in);
    l->activation.a_prime(aprime);

    Matrix *new_error = mul_matrices(aprime, output_error);

    free_matrix(aprime);

    return new_error;
}

void print_a_layer(ALayer *l)
{
    print_activation(l->activation_type);
}

void free_a_layer(ALayer *l)
{
    free_matrix(l->in);
    free_matrix(l->out);
}
