#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

typedef struct
{
    int w;
    int h;
} Dims;

typedef struct
{
    Dims dims;
    float *elems;

} Matrix;

Matrix *zeros_matrix(int w, int h);
Matrix *ones_matrix(int w, int h);

Matrix *to_matrix(float *, int, int, int);

void apply_c_matrix(Matrix *, float, float (*)(float, float));
void add_c_matrix(Matrix *, float);
void sub_c_matrix(Matrix *, float);
void mul_c_matrix(Matrix *, float);
void div_c_matrix(Matrix *, float);

void neg_matrix(Matrix *);

Matrix *trans_matrix(Matrix *);
Matrix *mul_matrix(Matrix *, Matrix *);

void print_matrix(Matrix *m);

void free_matrix(Matrix *);

#endif