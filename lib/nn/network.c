#include "network.h"

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"

Network *init_network(int input_size)
{
    Network *net = malloc(sizeof(Network));

    if (!net)
        return NULL;

    net->input_size = input_size;
    net->num_layers = 0;
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

    Layer *layer;

    if (net->num_layers == 0)
    {
        layer = init_layer(size, net->input_size);
    }
    else
    {
        layer = init_layer(size, net->layers[net->num_layers - 1]->size);
    }

    net->layers = realloc(net->layers, (net->num_layers + 1) * sizeof(Layer *));
    net->layers[net->num_layers] = layer;

    net->num_layers++;
}

void forward_network(Network *net, Matrix *x)
{
    for (int i = 0; i < net->num_layers; ++i)
    {
        forward_layer(net->layers[i], x);
        x = net->layers[i]->out;
    }
}

void print_network(Network *net)
{
    if (net->num_layers == 0)
    {
        return;
    }

    printf("x -> Layer(%d, %d)", net->layers[0]->input_size, net->layers[0]->size);

    for (int i = 1; i < net->num_layers; ++i)
    {
        printf(" -> Layer(%d, %d)", net->layers[i]->input_size, net->layers[i]->size);
    }

    printf(" -> y\n");
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
