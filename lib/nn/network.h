#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "layer.h"

typedef struct
{
    Layer **layers;
    int num_layers;

    int input_size;
} Network;

Network *init_network(int);
void print_network(Network *);

void add_layer_network(Network *, int);
void forward_network(Network *, Matrix *);
Matrix *output_network(Network *);

void free_network(Network *);

#endif