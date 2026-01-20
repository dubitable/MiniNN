#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "layer.h"
#include "../data/dataset.h"
#include "../metrics/loss.h"

typedef struct
{
    Layer **layers;
    int num_layers;

    int input_size;
    int output_size;

    LossFns loss;
} Network;

Network *init_network(int, int, Loss);

void print_network(Network *);
void free_network(Network *);

int check_network(Network *, DatasetSplit *);

void add_fc_layer_network(Network *, int);
void add_a_layer_network(Network *, Activation);

void forward_network(Network *, Matrix *);
void backward_network(Network *, Matrix *, float);

void fit_network(Network *, DatasetSplit *, float, float);

#endif