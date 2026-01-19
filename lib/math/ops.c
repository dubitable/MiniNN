#include "ops.h"

#include "math.h"

float op_add(float a, float b)
{
    return a + b;
}

float op_sub(float a, float b)
{
    return a - b;
}

float op_mul(float a, float b)
{
    return a * b;
}

float op_div(float a, float b)
{
    return a / b;
}

float op_sq(float a)
{
    return powf(a, 2);
}

float op_exp(float a)
{
    return expf(a);
}
