#include "network.h"

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "../data/dataset.h"

Network *init_network(NetworkOptions *options)
{
    Network *net = malloc(sizeof(Network));

    if (!net)
        return NULL;

    net->options = options;
    net->num_layers = 0;
    net->layers = NULL;

    return net;
}

Matrix *output_network(Network *net)
{
    if (net->num_layers == 0)
        return NULL;

    return net->layers[net->num_layers - 1]->out;
}

void add_layer_network(Network *net, int size)
{
}

void forward_network(Network *net, Matrix *x)
{
}

void backward_network(Network *net, Matrix *dL_dout, Matrix *input)
{
}

void train_network(Network *net, Dataset *dataset)
{
}

float get_loss(Network *net, Dataset *dataset)
{
    return 0;
}

void fit_network(Network *net, DatasetSplit *split, float epochs)
{
    for (int i = 0; i < epochs; ++i)
    {
        train_network(net, split->train);

        float risk = get_loss(net, split->val);
        printf("Epoch %d | val risk = %.3f\n", i + 1, risk);

        fflush(stdout);
    }

    printf("Summary | test risk = %.3f\n", get_loss(net, split->test));
}

void print_network(Network *net)
{
}

void free_network(Network *net)
{
    for (int i = 0; i < net->num_layers; ++i)
    {
        free_layer(net->layers[i]);
    }

    free(net->layers);
    free(net);
}
