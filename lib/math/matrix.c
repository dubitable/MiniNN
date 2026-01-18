#include "matrix.h"

#include <stdlib.h>
#include <stdio.h>

#include "ops.h"
#include "random.h"

Dims dims(int h, int w)
{
    Dims dims;

    dims.w = w;
    dims.h = h;

    return dims;
}

int rc_to_i(int r, int c, Dims dims)
{
    return r * dims.w + c;
}

Matrix *init_matrix(int h, int w)
{
    Matrix *m = malloc(sizeof(Matrix));

    if (!m)
        return NULL;

    m->dims = dims(h, w);

    float *elems = malloc(sizeof(float) * w * h);

    if (!elems)
        return NULL;

    m->elems = elems;

    return m;
}

Matrix *default_matrix(int h, int w, float def)
{
    Matrix *m = init_matrix(h, w);

    if (!m)
        return NULL;

    for (int i = 0; i < w * h; ++i)
    {
        m->elems[i] = def;
    }

    return m;
}

Matrix *zeros_matrix(int h, int w)
{
    return default_matrix(h, w, 0);
}

Matrix *ones_matrix(int h, int w)
{
    return default_matrix(h, w, 1);
}

Matrix *random_normal_matrix(int h, int w, float sd)
{
    Matrix *m = init_matrix(h, w);

    if (!m)
        return NULL;

    for (int i = 0; i < w * h; ++i)
    {
        m->elems[i] = irwin_hall() * sd;
    }

    return m;
}

Matrix *to_matrix(float *arr, int length, int h, int w)
{
    if (w * h != length)
    {
        printf("[ERR] incompatible dims, %d != %d x %d\n", length, h, w);
        return NULL;
    }

    Matrix *m = init_matrix(h, w);

    if (!m)
        return NULL;

    for (int i = 0; i < length; ++i)
    {
        m->elems[i] = arr[i];
    }

    return m;
}

Matrix *copy_matrix(Matrix *m)
{
    return to_matrix(m->elems, m->dims.h * m->dims.w, m->dims.h, m->dims.w);
}

void apply_matrix(Matrix *m, float (*op)(float))
{
    for (int i = 0; i < m->dims.w * m->dims.h; ++i)
    {
        m->elems[i] = op(m->elems[i]);
    }
}

void relu_matrix(Matrix *m)
{
    apply_matrix(m, &op_relu);
}

void apply_c_matrix(Matrix *m, float c, float (*op)(float, float))
{
    for (int i = 0; i < m->dims.w * m->dims.h; ++i)
    {
        m->elems[i] = op(m->elems[i], c);
    }
}

void add_c_matrix(Matrix *m, float c)
{
    apply_c_matrix(m, c, &op_add);
}

void sub_c_matrix(Matrix *m, float c)
{
    apply_c_matrix(m, c, &op_sub);
}

void mul_c_matrix(Matrix *m, float c)
{
    apply_c_matrix(m, c, &op_mul);
}

void div_c_matrix(Matrix *m, float c)
{
    apply_c_matrix(m, c, &op_div);
}

void neg_matrix(Matrix *m)
{
    mul_c_matrix(m, -1);
}

void add_ones_row_matrix(Matrix *m)
{
    int old_h = m->dims.h;
    int w = m->dims.w;
    int new_h = old_h + 1;

    m->elems = realloc(m->elems, sizeof(float) * new_h * w);
    m->dims.h = new_h;

    for (int j = 0; j < w; j++)
    {
        m->elems[rc_to_i(old_h, j, m->dims)] = 1.0f;
    }
}

Matrix *trans_matrix(Matrix *m)
{
    Matrix *out = zeros_matrix(m->dims.w, m->dims.h);

    for (int r = 0; r < m->dims.h; ++r)
    {
        for (int c = 0; c < m->dims.w; ++c)
        {
            out->elems[rc_to_i(c, r, out->dims)] = m->elems[rc_to_i(r, c, m->dims)];
        }
    }

    return out;
}

Matrix *mul_matrix(Matrix *a, Matrix *b)
{
    if (a->dims.w != b->dims.h)
    {
        printf("[ERR] incompatible dims, %d != %d\n", a->dims.w, b->dims.h);
        return NULL;
    }

    int n = a->dims.w;

    Matrix *out = zeros_matrix(a->dims.h, b->dims.w);

    for (int r = 0; r < a->dims.h; ++r)
    {
        for (int c = 0; c < b->dims.w; ++c)
        {
            for (int k = 0; k < n; ++k)
            {
                float a_i = a->elems[rc_to_i(r, k, a->dims)];
                float b_i = b->elems[rc_to_i(k, c, b->dims)];
                out->elems[rc_to_i(r, c, out->dims)] += a_i * b_i;
            }
        }
    }

    return out;
}

void print_matrix_prec(Matrix *m, int prec)
{
    for (int r = 0; r < m->dims.h; ++r)
    {
        printf("[");
        for (int c = 0; c < m->dims.w; ++c)
        {
            printf(" %.*f ", prec, m->elems[rc_to_i(r, c, m->dims)]);
        }
        printf("]\n");
    }
    printf("(%d, %d)\n", m->dims.h, m->dims.w);
}

const int DEFAULT_PRINT_PRECISION = 3;

void print_matrix(Matrix *m)
{
    print_matrix_prec(m, DEFAULT_PRINT_PRECISION);
}

void free_matrix(Matrix *m)
{
    free(m->elems);
    free(m);
}