#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "layer.h"
#include "../data/dataset.h"

typedef struct
{
    float lr;
    int input_size;
    int output_size;

    float (*loss)(Matrix *, Matrix *);
} NetworkOptions;

typedef struct
{
    Layer **layers;
    int num_layers;

    NetworkOptions *options;
} Network;

Network *init_network(NetworkOptions *);
void add_layer_network(Network *, int);

void train_network(Network *net, Dataset *dataset);
void fit_network(Network *net, DatasetSplit *split, float epochs);

void forward_network(Network *, Matrix *);
Matrix *output_network(Network *);

void print_network(Network *);
void free_network(Network *);

#endif