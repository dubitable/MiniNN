#ifndef ALAYER_H_INCLUDED
#define ALAYER_H_INCLUDED

#include "../../math/matrix.h"
#include "layertype.h"

typedef struct
{
    Matrix *in;
    Matrix *out;
    LayerType type;

    void (*activation)(Matrix *);
    void (*activation_prime)(Matrix *);
} ALayer;

ALayer *init_a_layer(void (*a)(Matrix *), void (*a_prime)(Matrix *));

void forward_a_layer(ALayer *, Matrix *);
Matrix *backward_a_layer(ALayer *, Matrix *);

void free_a_layer(ALayer *);

#endif