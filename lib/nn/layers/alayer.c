#include "alayer.h"

#include <stdlib.h>

ALayer *init_a_layer(void (*a)(Matrix *), void (*a_prime)(Matrix *))
{
    ALayer *l = malloc(sizeof(ALayer));

    if (!l)
        return NULL;

    l->type = LAYER_ACTIVATION;

    l->activation = a;
    l->activation_prime = a_prime;

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

    l->activation(l->out);
}

Matrix *backward_a_layer(ALayer *l, Matrix *output_error)
{
    Matrix *aprime = copy_matrix(l->in);
    l->activation_prime(aprime);

    Matrix *new_error = mul_matrices(aprime, output_error);

    free(aprime);

    return new_error;
}

void free_a_layer(ALayer *l)
{
    free_matrix(l->in);
    free_matrix(l->out);
}
