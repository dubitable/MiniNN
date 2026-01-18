#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "layer.h"
#include "../data/dataset.h"

typedef struct
{
    Layer **layers;
    int num_layers;

    int input_size;
    float lr;
} Network;

Network *init_network(int);
void print_network(Network *);

void train_network(Network *, Dataset *, float);
void train_pass_network(Network *, Dataset *);
void add_layer_network(Network *, int);
void forward_network(Network *, Matrix *);
Matrix *output_network(Network *);

void free_network(Network *);

#endif