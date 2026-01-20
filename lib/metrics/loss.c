#include "loss.h"

#include <math.h>
#include <stdlib.h>

#include "../math/matrix.h"

float mse_loss(Matrix *y_true, Matrix *y_est)
{
    Matrix *diff = sub_matrices(y_est, y_true);
    sq_matrix(diff);

    float sum = sum_matrix(diff);
    free(diff);
    return sum / (y_true->dims.h * y_true->dims.w);
}

Matrix *mse_prime_loss(Matrix *y_true, Matrix *y_est)
{
    Matrix *diff = sub_matrices(y_est, y_true);
    mul_c_matrix(diff, 2);

    div_c_matrix(diff, (y_true->dims.h * y_true->dims.w));

    return diff;
}

LossFns use_loss(Loss l)
{
    LossFns out;

    switch (l)
    {
    case LOSS_MSE:
        out.loss = &mse_loss;
        out.loss_prime = &mse_prime_loss;
        break;

    default:
        out.loss = NULL;
        out.loss_prime = NULL;
        break;
    }

    return out;
}