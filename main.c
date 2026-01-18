#include "lib/math/matrix.h"
#include "lib/nn/network.h"
#include "lib/data/dataset.h"
#include "lib/data/file.h"

#include <stdlib.h>
#include <time.h>

int main()
{
    srand(time(NULL));

    Dataset *parity = from_file_dataset("./examples/parity/parity.mini");

    if (!parity)
        return 1;

    DataSample sample = sample_dataset(parity);

    print_matrix(sample.x);
    print_matrix(sample.y);

    Network *net = init_network(parity->input_size);

    add_layer_network(net, 10);
    add_layer_network(net, 5);

    print_network(net);

    free_dataset(parity);
    free_network(net);
    return 0;
}