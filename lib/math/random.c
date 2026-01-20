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

int *generate_order(int count)
{
    int *order = malloc(sizeof(int) * count);
    if (!order)
        return NULL;

    for (int i = 0; i < count; i++)
        order[i] = i;

    return order;
}

void shuffle(int *order, int count)
{
    for (int i = count - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}