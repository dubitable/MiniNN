#include "layer.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../math/matrix.h"
#include "layers/fclayer.h"

void forward_layer(Layer *l, Matrix *x)
{
    switch (l->type)
    {
    case LAYER_FULLYCONNECTED:
        forward_fc_layer((FCLayer *)l, x);
        break;
    case LAYER_ACTIVATION:
        forward_a_layer((ALayer *)l, x);
        break;
    default:
        break;
    }
}

Matrix *backward_layer(Layer *l, Matrix *output_error, float lr)
{
    if (!l->out)
    {
        printf("[ERR] Not forward propagated yet.\n");
        return NULL;
    }

    switch (l->type)
    {
    case LAYER_FULLYCONNECTED:
        return backward_fc_layer((FCLayer *)l, output_error, lr);

    case LAYER_ACTIVATION:
        return backward_a_layer((ALayer *)l, output_error);

    default:
        break;
    }

    return NULL;
}

void print_layer(Layer *l)
{
    switch (l->type)
    {
    case LAYER_FULLYCONNECTED:
        print_fc_layer();
        break;

    case LAYER_ACTIVATION:
        print_a_layer((ALayer *)l);
        break;

    default:
        break;
    }
    printf("(%d, %d)", l->input_size, l->output_size);
}

int num_params_layer(Layer *l)
{
    switch (l->type)
    {
    case LAYER_FULLYCONNECTED:
        return num_params_fc_layer((FCLayer *)l);
        break;

    case LAYER_ACTIVATION:
        return 0;
        break;

    default:
        break;
    }
}

void free_layer(Layer *l)
{
    switch (l->type)
    {
    case LAYER_FULLYCONNECTED:
        free_fc_layer((FCLayer *)l);
        break;

    case LAYER_ACTIVATION:
        free_a_layer((ALayer *)l);
        break;

    default:
        break;
    }
}
