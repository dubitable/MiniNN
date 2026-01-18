gcc main.c \
    lib/math/matrix.c lib/math/ops.c lib/math/random.c \
    lib/data/dataset.c lib/data/file.c \
    lib/nn/layer.c lib/nn/network.c \
    -o main.exe \
    -Wall -Wextra -Werror