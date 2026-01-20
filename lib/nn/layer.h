#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "../math/matrix.h"
#include "layers/fclayer.h"
#include "layers/alayer.h"

typedef struct
{
    Matrix *in;
    Matrix *out;
    LayerType type;

    int input_size;
    int output_size;
} Layer;

void forward_layer(Layer *l, Matrix *x);
Matrix *backward_layer(Layer *, Matrix *, float);

void print_layer(Layer *l);

void free_layer(Layer *);

#endif