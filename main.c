#include "lib/nn/layer.h"
#include "lib/math/matrix.h"

int main()
{
    int input_size = 4;
    float x_arr[] = {2.0f, 3.0f, 4.0f, 5.0f};

    Matrix *x = to_matrix(x_arr, input_size, input_size, 1);

    print_matrix(x);

    Layer *l = init_layer(4, input_size);

    print_matrix(l->weights);

    Matrix *y = forward_layer(l, x);

    print_matrix(y);

    free_matrix(x);
    free_matrix(y);
    free_layer(l);
    return 0;
}