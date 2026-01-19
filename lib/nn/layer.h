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
} Layer;

FCLayer *init_fc_layer(int input_size, int output_size);
ALayer *init_a_layer(void (*a)(Matrix *), void (*a_prime)(Matrix *));

void forward_layer(Layer *l, Matrix *x);
Matrix *backward_layer(Layer *, Matrix *, float);

void free_layer(Layer *);

#endif