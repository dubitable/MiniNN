#include "lib/math/matrix.h"
#include "lib/nn/network.h"
#include "lib/data/dataset.h"
#include "lib/data/file.h"
#include "lib/nn/loss.h"

#include <stdlib.h>
#include <time.h>
#include <stdio.h>

int main()
{
    srand(time(NULL));

    Dataset *parity = from_file_dataset("./examples/parity/parity.mini");

    if (!parity)
        return 1;

    Network *net = init_network(parity->input_size);

    add_layer_network(net, 50);
    add_layer_network(net, 100);
    add_layer_network(net, parity->output_size);

    net->lr = 0.01;

    print_network(net);

    train_network(net, parity, 50);

    for (int i = 0; i < 5; ++i)
    {
        DataSample sample = sample_dataset(parity);

        forward_network(net, sample.x);

        Matrix *soft = softmax(output_network(net));

        if (soft->elems[0] > soft->elems[1])
        {
            printf("%f is even with confidence %.2f%%\n", sample.x->elems[0], soft->elems[0] * 100);
        }
        else
        {
            printf("%f is odd with confidence %.2f%%\n", sample.x->elems[0], soft->elems[1] * 100);
        }
        free_matrix(soft);
    }

    free_dataset(parity);
    free_network(net);
    return 0;
}