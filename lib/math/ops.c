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

float op_relu(float a)
{
    return a > 0 ? a : 0;
}

float op_reluder(float a)
{
    return a > 0 ? 1 : 0;
}

float op_exp(float a)
{
    return exp(a);
}
