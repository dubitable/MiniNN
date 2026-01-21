#ifndef OPS_H_INCLUDED
#define OPS_H_INCLUDED

float op_add(float, float);
float op_sub(float, float);
float op_mul(float, float);
float op_div(float, float);

float op_sq(float);
float op_exp(float);

float op_relu(float);
float op_relu_prime(float);
float op_tanh(float);
float op_tanh_prime(float);

#endif