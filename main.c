#include "lib/nn/network.h"
#include "lib/nn/layer.h"

#include "lib/metrics/activation.h"
#include "lib/metrics/loss.h"

#include "lib/nn/layers/fclayer.h"
#include "lib/nn/layers/alayer.h"

#include "lib/data/dataset.h"
#include "lib/data/file.h"

#include <stdlib.h>
#include <time.h>
#include <stdio.h>

int main()
{
    srand(20);

    Dataset *xor = from_file_dataset("./examples/xor/xor.mini");

    int input_size = xor->input_size;
    int output_size = xor->output_size;

    DatasetSplit *split = train_test_val(xor, 0.8, 0.1);

    Network *net = init_network(input_size, output_size, LOSS_MSE);

    add_fc_layer_network(net, 100);
    add_a_layer_network(net, ACTIVATION_RELU);
    add_fc_layer_network(net, 100);
    add_a_layer_network(net, ACTIVATION_RELU);
    add_fc_layer_network(net, output_size);

    int result = check_network(net, split);

    if (!result)
        return 1;

    print_network(net);

    fit_network(net, split, 10, 0.001f);

    Matrix *test1 = to_matrix((float[]){0, 0}, 2, 1, 2);
    forward_network(net, test1);
    print_matrix(net->layers[net->num_layers - 1]->out);
    free_matrix(test1);

    Matrix *test2 = to_matrix((float[]){0, 1}, 2, 1, 2);
    forward_network(net, test2);
    print_matrix(net->layers[net->num_layers - 1]->out);
    free_matrix(test2);

    Matrix *test3 = to_matrix((float[]){1, 0}, 2, 1, 2);
    forward_network(net, test3);
    print_matrix(net->layers[net->num_layers - 1]->out);
    free_matrix(test3);

    Matrix *test4 = to_matrix((float[]){1, 1}, 2, 1, 2);
    forward_network(net, test4);
    print_matrix(net->layers[net->num_layers - 1]->out);
    free_matrix(test4);

    free_datasetsplit(split);
    free_network(net);

    return 0;
}