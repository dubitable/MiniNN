#include "activation.h"

#include <stdlib.h>
#include <stdio.h>

#include "../math/matrix.h"
#include "../math/ops.h"

void relu_activation(Matrix *m)
{
    apply_matrix(m, &op_relu);
}

void relu_prime_activation(Matrix *m)
{
    apply_matrix(m, &op_relu_prime);
}

void tanh_activation(Matrix *m)
{
    apply_matrix(m, &op_tanh);
}

void tanh_prime_activation(Matrix *m)
{
    apply_matrix(m, &op_tanh_prime);
}

ActivationFns use_activation(Activation a)
{
    ActivationFns out;

    switch (a)
    {
    case ACTIVATION_RELU:
        out.a = &relu_activation;
        out.a_prime = &relu_prime_activation;
        break;

    case ACTIVATION_TANH:
        out.a = &tanh_activation;
        out.a_prime = &tanh_prime_activation;

        break;
    default:
        out.a = NULL;
        out.a_prime = NULL;
        break;
    }

    return out;
}

void print_activation(Activation a)
{
    switch (a)
    {
    case ACTIVATION_RELU:
        printf("ReLU");
        break;

    case ACTIVATION_TANH:
        printf("tanh");
        break;

    default:
        break;
    }
}