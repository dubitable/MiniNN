#include "random.h"

#include <stdlib.h>

// https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
float irwin_hall()
{
    float sum = 0.0f;
    for (int i = 0; i < 12; ++i)
    {
        sum += (float)rand() / (float)RAND_MAX;
    }
    return sum - 6.0f;
}