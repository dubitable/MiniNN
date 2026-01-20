#ifndef ALAYER_H_INCLUDED
#define ALAYER_H_INCLUDED

#include "../../math/matrix.h"
#include "../../metrics/activation.h"
#include "layertype.h"

typedef struct
{
    Matrix *in;
    Matrix *out;
    LayerType type;

    int input_size;
    int output_size;

    Activation activation_type;
    ActivationFns activation;
} ALayer;

ALayer *init_a_layer(int, int, Activation);

void forward_a_layer(ALayer *, Matrix *);
Matrix *backward_a_layer(ALayer *, Matrix *);

void print_a_layer(ALayer *);

void free_a_layer(ALayer *);

#endif