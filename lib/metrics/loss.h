#ifndef LOSS_H_INCLUDED
#define LOSS_H_INCLUDED

#include "../math/matrix.h"

float mse(Matrix *y_true, Matrix *y_est);
Matrix *mse_prime(Matrix *y_true, Matrix *y_est);

#endif