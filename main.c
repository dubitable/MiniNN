#include "lib/math/matrix.h"

int main()
{
    float arr_a[] = {1.0, 2.5, 3.0, 4.3, 3.2, 1.2, 3.4, 6.1};
    Matrix *a = to_matrix(arr_a, sizeof(arr_a) / sizeof(float), 2, 4);

    if (!a)
        return 1;

    print_matrix(a);

    Matrix *b = trans_matrix(a);

    if (!b)
        return 1;

    print_matrix(b);

    Matrix *c = mul_matrix(a, b);

    if (!c)
        return 1;

    print_matrix(c);

    free_matrix(a);
    free_matrix(b);
    free_matrix(c);

    return 0;
}