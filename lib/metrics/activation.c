#include "activation.h"

#include <stdlib.h>

#include "../math/matrix.h"

float op_relu(float a)
{
    return a > 0 ? a : 0;
}

float op_relu_prime(float a)
{
    return a > 0 ? 1 : 0;
}

void relu_activation(Matrix *m)
{
    apply_matrix(m, &op_relu);
}

void relu_prime_activation(Matrix *m)
{
    apply_matrix(m, &op_relu_prime);
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

    default:
        out.a = NULL;
        out.a_prime = NULL;
        break;
    }

    return out;
}