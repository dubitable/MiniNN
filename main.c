#include "lib/math/matrix.h"
#include "lib/nn/network.h"

int main()
{
    int input_size = 4;
    float x_arr[] = {2.0f, 3.0f, 4.0f, 5.0f};

    Matrix *x = to_matrix(x_arr, input_size, input_size, 1);

    Network *net = init_network(input_size);

    add_layer_network(net, 10);
    add_layer_network(net, 5);

    print_network(net);

    forward_network(net, x);

    free_matrix(x);
    free_network(net);
    return 0;
}