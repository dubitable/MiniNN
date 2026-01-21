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

float op_relu(float a)
{
    return a > 0 ? a : 0;
}

float op_relu_prime(float a)
{
    return a > 0 ? 1 : 0;
}

float op_tanh(float a)
{
    return tanhf(a);
}

float op_tanh_prime(float a)
{
    return 1 - op_sq(tanhf(a));
}
