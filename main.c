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
    srand(time(NULL));

    Dataset *dataset = from_file_dataset("./examples/xor/xor.mini");

    int input_size = dataset->input_size;
    int output_size = dataset->output_size;

    DatasetSplit *split = train_test_val(dataset, 0.8, 0.1, 1);

    Network *net = init_network(input_size, output_size, LOSS_CE);

    add_fc_layer_network(net, 100);
    add_a_layer_network(net, ACTIVATION_RELU);
    add_fc_layer_network(net, 100);
    add_a_layer_network(net, ACTIVATION_RELU);
    add_fc_layer_network(net, output_size);

    int result = check_network(net, split);

    if (!result)
        return 1;

    print_network(net);

    fit_network(net, split, 100, 0.01f, 1);

    free_datasetsplit(split);
    free_network(net);

    return 0;
}