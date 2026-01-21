#include "loss.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

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

void softmax(Matrix *m)
{
    float max = max_matrix(m); // numerical stability
    sub_c_matrix(m, max);

    exp_matrix(m);

    float sum = sum_matrix(m);
    div_c_matrix(m, sum);
}

float ce_loss(Matrix *y_true, Matrix *y_est)
{
    Matrix *soft = copy_matrix(y_est);
    softmax(soft);

    float loss = 0.0f;
    int n = y_true->dims.h * y_true->dims.w;

    for (int i = 0; i < n; i++)
    {
        loss -= y_true->elems[i] * logf(soft->elems[i] + 1e-8f);
    }

    free_matrix(soft);
    return loss / n;
}

Matrix *ce_prime_loss(Matrix *y_true, Matrix *y_est)
{
    Matrix *soft = copy_matrix(y_est);
    softmax(soft);

    Matrix *grad = sub_matrices(soft, y_true);

    int n = y_true->dims.h * y_true->dims.w;
    div_c_matrix(grad, n);

    free_matrix(soft);
    return grad;
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

    case LOSS_CE:
        out.loss = &ce_loss;
        out.loss_prime = &ce_prime_loss;
        break;

    default:
        out.loss = NULL;
        out.loss_prime = NULL;
        break;
    }

    return out;
}

void print_loss(Loss l)
{
    switch (l)
    {
    case LOSS_MSE:
        printf("MSE");
        break;

    case LOSS_CE:
        printf("Softmax + CE");
        break;
    default:
        break;
    }
}