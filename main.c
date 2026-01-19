#include "lib/nn/network.h"
#include "lib/nn/layer.h"
#include "lib/data/dataset.h"
#include "lib/data/file.h"
#include "lib/metrics/activation.h"
#include "lib/metrics/loss.h"

#include <stdlib.h>
#include <time.h>
#include <stdio.h>

int main()
{
    srand(time(NULL));

    Dataset *xor = from_file_dataset("./examples/xor/xor.mini");

    DatasetSplit *split = train_test_val(xor, 0.8, 0.1);

    NetworkOptions options = {
        .input_size = split->train->input_size,
        .output_size = split->train->output_size,
        .lr = 0.001};

    DataSample sample = sample_dataset(split->train);

    print_matrix(sample.x);
    print_matrix(sample.y);

    FCLayer *l = init_fc_layer(2, 1);
    print_matrix(l->weights);
    print_matrix(l->bias);

    forward_layer((Layer *)l, sample.x);
    print_matrix(l->out);

    backward_layer((Layer *)l, mse_prime(sample.y, l->out), 1);
    print_matrix(l->weights);

    free_datasetsplit(split);
    free_layer((Layer *)l);

    return 0;
}