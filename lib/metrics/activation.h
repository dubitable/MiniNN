#ifndef ACTIVATION_H_INCLUDED
#define ACTIVATION_H_INCLUDED

#include "../math/matrix.h"

void relu_activation(Matrix *);
void relu_prime_activation(Matrix *);

typedef enum
{
    ACTIVATION_RELU,
    ACTIVATION_TANH
} Activation;

typedef struct
{
    void (*a)(Matrix *);
    void (*a_prime)(Matrix *);
} ActivationFns;

ActivationFns use_activation(Activation);

void print_activation(Activation);

#endif