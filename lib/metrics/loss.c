#include "loss.h"

#include <math.h>
#include <stdlib.h>

#include "../math/matrix.h"

float mse(Matrix *y_true, Matrix *y_est)
{
    Matrix *diff = sub_matrices(y_est, y_true);
    sq_matrix(diff);

    float sum = sum_matrix(diff);
    free(diff);
    return sum / (y_true->dims.h * y_true->dims.w);
}

Matrix *mse_prime(Matrix *y_true, Matrix *y_est)
{
    Matrix *diff = sub_matrices(y_est, y_true);
    mul_c_matrix(diff, 2);
    div_c_matrix(diff, (y_true->dims.h * y_true->dims.w));

    return diff;
}