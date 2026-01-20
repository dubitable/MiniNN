#ifndef LOSS_H_INCLUDED
#define LOSS_H_INCLUDED

#include "../math/matrix.h"

float mse(Matrix *, Matrix *);
Matrix *mse_prime(Matrix *, Matrix *);

typedef enum
{
    LOSS_MSE
} Loss;

extern char *loss_names[];

typedef struct
{
    float (*loss)(Matrix *, Matrix *);
    Matrix *(*loss_prime)(Matrix *, Matrix *);
} LossFns;

LossFns use_loss(Loss);

#endif