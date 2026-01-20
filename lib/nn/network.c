#include "network.h"

#include <stdlib.h>
#include <stdio.h>

#include "layer.h"
#include "layers/fclayer.h"
#include "layers/alayer.h"
#include "../data/dataset.h"
#include "../metrics/loss.h"

Network *init_network(int input_size, int output_size, Loss loss)
{
    Network *net = malloc(sizeof(Network));

    if (!net)
        return NULL;

    net->input_size = input_size;
    net->output_size = output_size;

    net->num_layers = 0;
    net->layers = NULL;

    net->loss = use_loss(loss);
    net->loss_type = loss;

    return net;
}

Matrix *output_network(Network *net)
{
    return net->layers[net->num_layers - 1]->out;
}

void add_layer_network(Network *net, Layer *layer)
{
    net->num_layers++;
    net->layers = realloc(net->layers, net->num_layers * sizeof(Layer *));
    net->layers[net->num_layers - 1] = layer;
}

void add_fc_layer_network(Network *net, int size)
{
    FCLayer *layer;
    if (net->num_layers == 0)
    {
        layer = init_fc_layer(net->input_size, size);
    }
    else
    {
        layer = init_fc_layer(net->layers[net->num_layers - 1]->output_size, size);
    }

    add_layer_network(net, (Layer *)layer);
}

void add_a_layer_network(Network *net, Activation a)
{
    ALayer *layer;
    if (net->num_layers == 0)
    {
        int size = net->input_size;
        layer = init_a_layer(size, size, a);
    }
    else
    {
        int size = net->layers[net->num_layers - 1]->output_size;
        layer = init_a_layer(size, size, a);
    }

    add_layer_network(net, (Layer *)layer);
}

void forward_network(Network *net, Matrix *x)
{
    for (int i = 0; i < net->num_layers; ++i)
    {
        forward_layer(net->layers[i], x);
        x = net->layers[i]->out;
    }
}

void backward_network(Network *net, Matrix *y_true, float lr)
{
    Matrix *output_error = net->loss.loss_prime(y_true, output_network(net));

    for (int i = net->num_layers - 1; i >= 0; --i)
    {
        Matrix *in_error = backward_layer(net->layers[i], output_error, lr);
        free_matrix(output_error);
        output_error = in_error;
    }
}

void train_network(Network *net, Dataset *dataset, float lr)
{
    for (int i = 0; i < dataset->count; ++i)
    {
        forward_network(net, dataset->x[i]);
        backward_network(net, dataset->y[i], lr);
    }
}

float get_risk(Network *net, Dataset *dataset)
{
    float risk = 0;
    for (int i = 0; i < dataset->count; ++i)
    {
        forward_network(net, dataset->x[i]);
        Matrix *y_est = output_network(net);

        risk += net->loss.loss(dataset->y[i], y_est);
    }
    return risk / dataset->count;
}

void fit_network(Network *net, DatasetSplit *split, float epochs, float lr)
{
    for (int i = 0; i < epochs; ++i)
    {
        train_network(net, split->train, lr);

        float train_risk = get_risk(net, split->train);
        float val_risk = get_risk(net, split->val);

        printf("Epoch %d | train risk = %.3f | val risk = %.3f\n", i + 1, train_risk, val_risk);

        fflush(stdout);
    }

    printf("Summary | test risk = %.3f\n", get_risk(net, split->test));
}

int check_network(Network *net, DatasetSplit *split)
{
    if (net->num_layers == 0)
    {
        printf("[ERR] Empty network.\n");
        return 0;
    }

    if (net->layers[net->num_layers - 1]->type == LAYER_ACTIVATION)
    {
        printf("[ERR] Last layer cannot be an activation layer.\n");
        return 0;
    }

    if (net->layers[0]->input_size != split->train->input_size)
    {
        printf("[ERR] First layer must be of the same size as the data input size.\n");
        return 0;
    }

    if (net->layers[net->num_layers - 1]->output_size != split->train->output_size)
    {
        printf("[ERR] Last layer must be of the same size as the data output size.\n");
        return 0;
    }

    if (net->loss.loss == NULL || net->loss.loss_prime == NULL)
    {
        printf("[ERR] Unspecified loss function.\n");
        return 0;
    }

    return 1;
}

int num_params_network(Network *net)
{
    int num_params = 0;
    for (int i = 0; i < net->output_size; ++i)
    {
        num_params += num_params_layer(net->layers[i]);
    }
    return num_params;
}

void print_network(Network *net)
{
    if (net->num_layers == 0)
    {
        printf("Empty network.\n");
        return;
    }

    int params = num_params_network(net);
    printf("Network Info\n");
    printf("|- %d params (~%d B)\n", params, (int)(params * sizeof(float)));
    printf("|- Loss function: %s\n", loss_names[net->loss_type]);
    printf("|- x(%d, %d) -> ", 1, net->input_size);
    print_layer(net->layers[0]);
    for (int i = 1; i < net->num_layers; ++i)
    {
        printf(" -> ");
        print_layer(net->layers[i]);
    }

    printf(" -> y(%d, %d)\n", net->output_size, 1);
}

void free_network(Network *net)
{
    for (int i = 0; i < net->num_layers; ++i)
    {
        free_layer(net->layers[i]);
    }

    free(net->layers);
    free(net);
}
