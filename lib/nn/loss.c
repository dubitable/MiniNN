#include "loss.h"

#include "../math/matrix.h"

Matrix *softmax(Matrix *y)
{
    Matrix *out = copy_matrix(y);

    exp_matrix(out);
    float sum = sum_matrix(out);
    div_c_matrix(out, sum);

    return out;
}
