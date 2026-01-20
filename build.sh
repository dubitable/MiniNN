gcc main.c \
    lib/math/matrix.c lib/math/ops.c lib/math/random.c \
    lib/data/dataset.c lib/data/file.c \
    lib/metrics/activation.c lib/metrics/loss.c \
    lib/nn/layer.c lib/nn/network.c \
        lib/nn/layers/fclayer.c lib/nn/layers/alayer.c \
    -o main.exe
    #-Wall -Wextra -Werror