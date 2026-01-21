#ifndef LOSS_H_INCLUDED
#define LOSS_H_INCLUDED

#include "../math/matrix.h"

float mse(Matrix *, Matrix *);
Matrix *mse_prime(Matrix *, Matrix *);

typedef enum
{
    LOSS_MSE,
    LOSS_CE
} Loss;

typedef struct
{
    float (*loss)(Matrix *, Matrix *);
    Matrix *(*loss_prime)(Matrix *, Matrix *);
} LossFns;

LossFns use_loss(Loss);

void print_loss(Loss);

#endif