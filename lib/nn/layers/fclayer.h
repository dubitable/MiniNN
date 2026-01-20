#ifndef FCLAYER_H_INCLUDED
#define FCLAYER_H_INCLUDED

#include "../../math/matrix.h"
#include "layertype.h"

typedef struct
{
    Matrix *in;
    Matrix *out;
    LayerType type;

    int input_size;
    int output_size;

    Matrix *weights;
    Matrix *bias;
} FCLayer;

FCLayer *init_fc_layer(int, int);

void forward_fc_layer(FCLayer *, Matrix *);
Matrix *backward_fc_layer(FCLayer *, Matrix *, float);

void print_fc_layer();

void free_fc_layer(FCLayer *);

#endif
