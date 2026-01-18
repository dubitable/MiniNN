#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include "../math/matrix.h"

typedef struct
{
    int size;
    int input_size;

    Matrix *weights;
    Matrix *out;
    Matrix *grad_weights;
} Layer;

Layer *init_layer(int, int);

void forward_layer(Layer *, Matrix *);
Matrix *backward_layer(Layer *, Matrix *, Matrix *);
void update_layer(Layer *, float);

void free_layer(Layer *);

#endif