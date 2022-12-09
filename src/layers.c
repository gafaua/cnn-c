#include "layers.h"
#define LR 0.00001

// Y must be 0 init
void convolution(Square X, Square W, Square Y) {
    for(int i=0; i < Y.size; i++)
        for(int j=0; j < Y.size; j++) {
            float sum = 0.0;
            for(int k=0; k < W.size; k++)
                for(int l=0; l < W.size; l++) 
                    sum += W.mat[k][l] * X.mat[k+i][l+j];
            Y.mat[i][j] += sum;
        }
}

void max_pool_conv_grad(Square X, Square Y, int* mem, int size) {
    for(int i=0; i < Y.size; i++)
        for(int j=0; j < Y.size; j++) {
            float max = -INFINITY;
            int mem_pos = j * 2 + i * Y.size;
            int ii = i * size;
            int jj = j * size;
            for(int k=0; k < size; k++)
                for(int l=0; l < size; l++)
                    if (X.mat[ii+k][jj+l] > max) {
                        max = X.mat[ii+k][jj+l];
                        mem[mem_pos] = ii+k;
                        mem[mem_pos + 1] = jj+l;
                    }
            Y.mat[i][j] = max;
        }
}

void max_pool_conv(Square X, Square Y, int size) {
    for(int i=0; i < Y.size; i+=size)
        for(int j=0; j < Y.size; j+=size) {
            float max = -INFINITY;
            int ii = i * size;
            int jj = j * size;
            for(int k=0; k < size; k++)
                for(int l=0; l < size; l++)
                    if (X.mat[k+ii][l+jj] > max)
                        max = X.mat[k+ii][l+jj];
            Y.mat[i][j] = max;
        }
}

void deconvolution(Square dY, Square X, Square dX, Square W, Square dW) {
    for(int i=0; i < dY.size; i++)
        for(int j=0; j < dY.size; j++) {
            float dYElem = dY.mat[i][j];
            for(int k=0; k < W.size; k++)
                for(int l=0; l < W.size; l++) {
                    dW.mat[k][l] += X.mat[k+i][l+j] * dYElem;
                    dX.mat[k+i][l+j] += W.mat[k][l] * dYElem;
                }
        }
}

// @brief Basic convolution, stride of 1, no padding (or included in X), only squares
// @param layer: Convolutionnal layer, weights of dim [out_channels, in_channels, k_size, k_size]
// @param input: Data2D, dim [batch, in_channels, size, size]
// @returns Data2D, dim [batch, out_channels, size, size]
Data2D* conv_forward(ConvLayer* layer, Data2D* input) {
    assert(input->c == layer->in && "The input must have the same number of channels as the input channels of this layer");

    if (layer->dW != NULL) {
        // Save inputs in layer->X
        if (layer->X != NULL) {
            DestroyData2D(layer->X);
        }

        layer->X = input;
    }

    int output_size = get_output_size(input->size, layer->size);
    Data2D* output = CreateData2DZeros(output_size, input->b, layer->out);
    int i,j,k;

    #pragma omp parallel for private(i,j,k)
    for (i = 0; i < output->b; i++) {
        Square* in = input->data[i];
        for (j = 0; j < output->c; j++) {
            Square* kernels = layer->w[j];
            Square out = output->data[i][j];
            for (k = 0; k < input->c; k++)
                convolution(in[k], kernels[k], out);
            //add_to_square(out, layer->b[j]);
        }
    }

    return output;
}

// @param dY: Data2D, dim [batch, out_channels, size, size]
// @param X: Data2D, dim [batch, in_channels, size, size]
// @param layer: Convolutionnal layer, weights of dim [out_channels, in_channels, k_size, k_size]
// @returns dX: Data2D, dim [batch, in_channels, size, size]
// @returns layer.dW: dim [out_channels, in_channels, k_size, k_size]
Data2D* conv_backward(ConvLayer* layer, Data2D* dY, float lr) {
    assert(layer->dW != NULL && "This convolutional layer wasn't initialized with the with_gradient flag");
    assert(layer->out == dY->c && "The gradient from the next layer doesn't have the same number of channels as the output channels of this convolutional layer");
    assert(layer->in == layer->X->c && "The input tensor given doesn't have the same number of channels as the input channels of this convolutional layer");

    int i,j,k;

    // Clear gradients
    Data2D* dX = CreateData2DZeros(layer->X->size, layer->X->b, layer->X->c);
    for (i = 0; i < layer->out; i++) {
        layer->db[i] = 0.0;
        for (j = 0; j < layer->in; j++)
            init_square(layer->dW[i][j], 0.0);
    }

    // Compute new gradients
    #pragma omp parallel for private(i,j,k)
    for (k = 0; k < layer->X->b; k++) {
        Square* dYb = dY->data[k];
        Square* Xb = layer->X->data[k];
        Square* dXb = dX->data[k];
        for (i = 0; i < layer->out; i++) {
            Square* kernels = layer->w[i];
            Square* dWOut = layer->dW[i];
            for (j = 0; j < layer->in; j++) {
                deconvolution(dYb[i], Xb[j], dXb[j], kernels[j], dWOut[j]);
                //layer->db[i] += sum_square(dWOut[j]);
            }
        }
    }

    // Learn from the new gradients
    LearnConvLayer(layer, lr);

    return dX;
}

/*
    @param layer.w: [output_features, input_features] 
    @param inputs: [batch_size, input_features]
    @returns outputs: [batch_size, output_features] */
Data1D* linear_forward(LinearLayer* layer, Data1D* input) {
    if (layer->in != input->n) {
        printf("[ERROR] Invalid size of input data for linear layer, layer: %d, input: %d\n", layer->in, input->n);
        exit(1);
    }
    
    if (layer->dW != NULL) {
        // Save inputs in layer.X
        if (layer->X != NULL) {
            DestroyData1D(layer->X);
        }

        layer->X = input;
    }

    Data1D* outputs = CreateData1D(layer->out, input->b);
    init_matrix(outputs->mat, 0.0, outputs->b, outputs->n);
    
    matrix_mul_2d_T2(input->mat, layer->w, outputs->mat, input->b, layer->in, layer->out);

    #pragma omp parallel for
    for (int i = 0; i < input->b; i++)
        for (int j = 0; j < layer->out; j++)
            outputs->mat[i][j] += layer->b[j];

    return outputs;
}

// @param dY: [batch_size, output_features]
// @param  X: [batch_size, input_features]
// @param layer.w: [output_features, input_features]
// @returns dX: [batch_size, input_features]
// @returns layer.dW: [output_features, input_features]
Data1D* linear_backward(LinearLayer* layer, Data1D* dY, float lr) {
    assert(layer->dW != NULL && "This linear layer wasn't initialized with the with_gradient flag");
    assert(dY->n == layer->out && "The gradient from the next layer has not the same number of features than this layer");
    assert(dY->b == layer->X->b && "The gradient from the next layer is not computed for the same number of batch than the input of this layer");

    Data1D* dX = CreateData1D(layer->X->n, layer->X->b);
    
    // Zero grad
    init_matrix(dX->mat, 0.0, dX->b, dX->n);
    for (int i = 0; i < layer->out; i++) {
        layer->db[i] = 0.0;
        for (int j = 0; j < layer->in; j++)
            layer->dW[i][j] = 0.0;
    }

    // dX = dY * layer.w
    matrix_mul_2d(dY->mat, layer->w, dX->mat, dX->b, layer->out, dX->n);
    // dW = dY.T * X
    matrix_mul_2d_T1(dY->mat, layer->X->mat, layer->dW, layer->out, dY->b, layer->in);
    // db = ones * dY
    #pragma omp parallel for
    for (int i = 0; i < dY->b; i++)
        for (int j = 0; j < dY->n; j++)
            layer->db[j] += dY->mat[i][j];

    // Learn from the new gradients
    LearnLinearLayer(layer, lr);

    return dX;
}

Data2D* max_pool_forward(MaxPoolLayer* layer, Data2D* input) {
    int output_size = input->size/layer->size;
    Data2D* output = CreateData2D(output_size, input->b, input->c);

    if (layer->with_gradient) {
        if (layer->mem != NULL) {
            free(layer->mem);
        }

        layer->mem = (int*) malloc(sizeof(int) * 2 * output_size * output_size * input->c * input->b);
        for (int i = 0; i < input->b; i++) {
            int ii = i * 2 * output_size * output_size * input->c;
            for (int j = 0; j < input->c; j++) {
                int jj = j * 2 * output_size * output_size;
                max_pool_conv_grad(input->data[i][j], output->data[i][j], &layer->mem[ii+jj], layer->size);
            }
        }
    }
    else {
        for (int i = 0; i < input->b; i++) {
            for (int j = 0; j < input->c; j++) {
                max_pool_conv(input->data[i][j], output->data[i][j], layer->size);
            }
        }
    }

    return output;
}

Data2D* max_pool_backward(MaxPoolLayer* layer, Data2D* dY) {
    assert(layer->with_gradient && "This Max Pool Layer needs to be gradient enabled to perform backward pass");

    Data2D* dX = CreateData2DZeros(dY->size * layer->size, dY->b, dY->c);

    for (int i = 0; i < dY->b; i++) {
        int ii = i * 2 * dY->size * dY->size * dY->c;
        for (int j = 0; j < dY->c; j++) {
            int jj = j * 2 * dY->size * dY->size;
            for (int k = 0; k < dY->size; k++) {
                for (int l = 0; l < dY->size; l++) {
                    int kk = ii + jj + k * dY->size + l * 2;
                    dX->data[i][j].mat[layer->mem[kk]][layer->mem[kk+1]] = dY->data[i][j].mat[k][l];
                    kk += 2;
                }
            }
        }
    }

    return dX;
}

Data2D* tanh_2d_forward(Activation2DLayer* layer, Data2D* input) {
    Data2D* output = CreateData2D(input->size, input->b, input->c);

    if (layer->with_gradient) {
        if (layer->X == NULL) {
            layer->X = CreateData2D(input->size, input->b, input->c);
        }
        else if (layer->X != NULL && (layer->X->b != input->b || layer->X->c != input->c || layer->X->size != input->size)) {
            DestroyData2D(layer->X);
            layer->X = CreateData2D(input->size, input->b, input->c);
        }

        #pragma omp parallel for
        for (int i = 0; i < input->b; i++)
            for (int j = 0; j < input->c; j++) 
                for (int k = 0; k < input->size; k++) 
                    for (int l = 0; l < input->size; l++)
                        output->data[i][j].mat[k][l] = layer->X->data[i][j].mat[k][l] = tanhf(input->data[i][j].mat[k][l]);
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < input->b; i++)
            for (int j = 0; j < input->c; j++) 
                for (int k = 0; k < input->size; k++) 
                    for (int l = 0; l < input->size; l++) 
                        output->data[i][j].mat[k][l] = tanhf(input->data[i][j].mat[k][l]);
    }

    return output;
}

Data2D* tanh_2d_backward(Activation2DLayer* layer, Data2D* dY) {
    assert(layer->with_gradient && "Can't perform backward pass on this tanh layer without gradient");
    
    #pragma omp parallel for
    for (int i = 0; i < dY->b; i++)
        for (int j = 0; j < dY->c; j++) 
            for (int k = 0; k < dY->size; k++) 
                for (int l = 0; l < dY->size; l++) {
                    dY->data[i][j].mat[k][l] *= (1 - powf(layer->X->data[i][j].mat[k][l], 2));
                }

    return dY;
}

Data1D* tanh_1d_forward(Activation1DLayer* layer, Data1D* input) {
    Data1D* output = CreateData1D(input->n, input->b);

    if (layer->with_gradient) {
        if (layer->X == NULL) 
            layer->X = CreateData1D(input->n, input->b);
        else if (layer->X != NULL && (layer->X->b != input->b || layer->X->n != input->n)) {
            DestroyData1D(layer->X);
            layer->X = CreateData1D(input->n, input->b);
        }

        #pragma omp parallel for
        for (int i = 0; i < input->b; i++)
            for (int j = 0; j < input->n; j++)
                output->mat[i][j] = layer->X->mat[i][j] = tanhf(input->mat[i][j]);
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < input->b; i++)
            for (int j = 0; j < input->n; j++) 
                output->mat[i][j] = tanhf(input->mat[i][j]);
    }

    return output;
}

Data1D* tanh_1d_backward(Activation1DLayer* layer, Data1D* dY) {
    assert(layer->with_gradient && "Can't perform backward pass on tanh relu layer without gradient");

    #pragma omp parallel for
    for (int i = 0; i < dY->b; i++)
        for (int j = 0; j < dY->n; j++) 
            dY->mat[i][j] *= (1 - powf(layer->X->mat[i][j], 2));

    return dY;
}

Data2D* relu_2d_forward(Activation2DLayer* layer, Data2D* input) {
    Data2D* output = CreateData2D(input->size, input->b, input->c);

    if (layer->with_gradient) {
        if (layer->X == NULL) {
            layer->X = CreateData2D(input->size, input->b, input->c);
        }
        else if (layer->X != NULL && (layer->X->b != input->b || layer->X->c != input->c || layer->X->size != input->size)) {
            DestroyData2D(layer->X);
            layer->X = CreateData2D(input->size, input->b, input->c);
        }

        #pragma omp parallel for
        for (int i = 0; i < input->b; i++)
            for (int j = 0; j < input->c; j++) 
                for (int k = 0; k < input->size; k++) 
                    for (int l = 0; l < input->size; l++) {
                        if (input->data[i][j].mat[k][l] > 0) {
                            output->data[i][j].mat[k][l] = input->data[i][j].mat[k][l];
                            layer->X->data[i][j].mat[k][l] = 1.0;
                        }
                        else {
                            output->data[i][j].mat[k][l] = 0.0001*input->data[i][j].mat[k][l];
                            layer->X->data[i][j].mat[k][l] = 0.0001;
                            //output->data[i][j].mat[k][l] = 0.0;
                            //layer->X->data[i][j].mat[k][l] = 0.0;
                        }
                    }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < input->b; i++)
            for (int j = 0; j < input->c; j++)
                for (int k = 0; k < input->size; k++)
                    for (int l = 0; l < input->size; l++) {
                        output->data[i][j].mat[k][l] = input->data[i][j].mat[k][l] > 0 ?
                                                       input->data[i][j].mat[k][l] :
                                                       0.0001*input->data[i][j].mat[k][l];
                    }
    }
    
    return output;
}

Data2D* relu_2d_backward(Activation2DLayer* layer, Data2D* dY) {
    assert(layer->with_gradient && "Can't perform backward pass on this relu layer without gradient");
    
    #pragma omp parallel for
    for (int i = 0; i < dY->b; i++)
        for (int j = 0; j < dY->c; j++) 
            for (int k = 0; k < dY->size; k++) 
                for (int l = 0; l < dY->size; l++)
                    dY->data[i][j].mat[k][l] *= layer->X->data[i][j].mat[k][l];
    
    return dY;
}

Data1D* relu_1d_forward(Activation1DLayer* layer, Data1D* input) {
    Data1D* output = CreateData1D(input->n, input->b);

    if (layer->with_gradient) {
        if (layer->X == NULL) 
            layer->X = CreateData1D(input->n, input->b);
        else if (layer->X != NULL && (layer->X->b != input->b || layer->X->n != input->n)) {
            DestroyData1D(layer->X);
            layer->X = CreateData1D(input->n, input->b);
        }

        #pragma omp parallel for
        for (int i = 0; i < input->b; i++)
            for (int j = 0; j < input->n; j++) 
                if (input->mat[i][j] > 0) {
                    output->mat[i][j] = input->mat[i][j];
                    layer->X->mat[i][j] = 1.0;
                } 
                else {
                    output->mat[i][j] = 0.0001 * input->mat[i][j];
                    layer->X->mat[i][j] = 0.0001;
                }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < input->b; i++)
            for (int j = 0; j < input->n; j++) 
                output->mat[i][j] = input->mat[i][j] > 0 ? 
                                    input->mat[i][j] : 
                                    0.0001 * input->mat[i][j];
    }

    return output;
}

Data1D* relu_1d_backward(Activation1DLayer* layer, Data1D* dY) {
    assert(layer->with_gradient && "Can't perform backward pass on this relu layer without gradient");

    #pragma omp parallel for
    for (int i = 0; i < dY->b; i++)
        for (int j = 0; j < dY->n; j++) 
            dY->mat[i][j] *= layer->X->mat[i][j];
    
    return dY;
}


float ReLU(float val) {
    if (val <= 0.0) return 0.0;
    else return val;
}

float ReLU_backward(float val) {
    if (val <= 0.0) return 0.0;
    else return 1.0;
}

float Identity(float val) {
    return val;
}

// @brief Computes the LogSoftmax of y_hat + Negative LogLikelihood with y
// @param y_hat: output predictions of size [batch, num_classes]
// @param y: indices of ground truth values of size [batch]
// @returns LossResult containing loss value and 
LossResult CrossEntropy(Network* net, Data1D* y_hat, int* y) {
    double sum, tmp, pred, max;
    int idx;
    LossResult result;
    result.dL = CreateData1D(y_hat->n, y_hat->b);
    init_matrix(result.dL->mat, 0.0, result.dL->b, result.dL->n);
    result.value = 0.0;
    result.accuracy = 0.0;
    float eps = 1e-7;
    float wd_factor = 0.0001;

    // print_data1d(y_hat);
    // Compute softmax in dL
    float weight_norm = GetLayersNorm(net);

    for (int i = 0; i < y_hat->b; i++) {
        sum = eps;
        for (int j = 0; j < y_hat->n; j++) {
            result.dL->mat[i][j] = exp(y_hat->mat[i][j]);
            sum += result.dL->mat[i][j];
        }
        //printf("Softmax b4: \n");
        //print_data1d(result.dL);

        for (int j = 0; j < y_hat->n; j++)
            result.dL->mat[i][j] /= sum;

        result.value -= logf(result.dL->mat[i][y[i]]+eps);
        if (argmax_vector(result.dL->mat[i], y_hat->n) == y[i])
            result.accuracy++;
        //printf("Softmax: \n");
        result.dL->mat[i][y[i]] -= 1.0;
        // for (int j = 0; j < y_hat->n; j++)
        //     result.dL->mat[i][j] *= 2 * weight_norm * wd_factor;
    }

    //print_data1d(result.dL);

    result.accuracy = result.accuracy / y_hat->b;
    result.value = result.value / y_hat->b;// * weight_norm * wd_factor / y_hat->b;
    return result;
}

// @param d: Data2D of shape [batch, channels, size, size]
// @returns d_: Data1D of shape [batch, channels*size*size]
Data1D* flatten(Data2D* d2) {
    Data1D* d1 = CreateData1D(d2->c*d2->size*d2->size, d2->b);

    #pragma omp parallel for
    for (int k = 0; k < d2->b; k++) {
        Square* db2 = d2->data[k];
        float* db1 = d1->mat[k];
        for (int l = 0; l < d2->c; l++){
            float** curr = db2[l].mat;
            int c_shift = l * d2->size * d2->size;
            for (int i = 0; i < d2->size; i++) {
                int vpos = i * d2->size;
                for (int j = 0; j < d2->size; j++)
                    db1[c_shift + vpos + j] = curr[i][j];
            }
        }
    }

    return d1;
}

// @param d: Data1D of shape [batch, size]
// @param channels: number of channels to create in the output tensor
// @returns d_: Data2D of shape [batch, channels, sqrt(size/channels), sqrt(size/channels)]
Data2D* unflatten(Data1D* d1, int channels) {
    int size = (int)sqrt((double) (d1->n/channels));
    assert(pow((double) size, 2)*channels == (double) d1->n && "Can't unflatten data with non square number of features with this number of channels");

    Data2D* d2 = CreateData2D(size, d1->b, channels);

    #pragma omp parallel for
    for (int k = 0; k < d2->b; k++) {
        Square* db2 = d2->data[k];
        float* db1 = d1->mat[k];
        for (int l = 0; l < d2->c; l++){
            float** curr = db2[l].mat;
            int c_shift = l * d2->size * d2->size;
            for (int i = 0; i < d2->size; i++) {
                int vpos = i * d2->size;
                for (int j = 0; j < d2->size; j++)
                    curr[i][j] = db1[c_shift + vpos + j];
            }
        }
    }

    return d2;
}

ViewLayer* CreateFlattenLayer(int channels) {
    ViewLayer* l = (ViewLayer*) malloc(sizeof(ViewLayer));
    l->node.type = Flatten;
    l->channels = channels;
    return l;
}

ViewLayer* CreateUnflattenLayer(int channels) {
    ViewLayer* l = (ViewLayer*) malloc(sizeof(ViewLayer));
    l->node.type = Unflatten;
    l->channels = channels;
    return l;
}

void DestroyViewLayer(ViewLayer* node) {
    free(node);
}

MaxPoolLayer* CreateMaxPoolLayer(int size, int with_gradient) {
    MaxPoolLayer* l = (MaxPoolLayer*) malloc(sizeof(MaxPoolLayer));
    l->size = size;
    l->node.type = MaxPool;
    l->mem = NULL;
    l->with_gradient = with_gradient;

    return l;
}

void DestroyMaxPoolLayer(MaxPoolLayer* layer) {
    if (layer->mem != NULL) free(layer->mem);
    free(layer);
}

Activation1DLayer* CreateReLU1DLayer(int with_gradient) {
    Activation1DLayer* l = (Activation1DLayer*) malloc(sizeof(Activation1DLayer));
    l->node.type = ReLU1D;
    l->X = NULL;
    l->with_gradient = with_gradient;

    return l;
}

Activation1DLayer* CreateTanh1DLayer(int with_gradient) {
    Activation1DLayer* l = (Activation1DLayer*) malloc(sizeof(Activation1DLayer));
    l->node.type = Tanh1D;
    l->X = NULL;
    l->with_gradient = with_gradient;

    return l;
}

void DestroyActivation1DLayer(Activation1DLayer* layer) {
    if (layer->X != NULL) DestroyData1D(layer->X);
    free(layer);
}

Activation2DLayer* CreateReLU2DLayer(int with_gradient) {
    Activation2DLayer* l = (Activation2DLayer*) malloc(sizeof(Activation2DLayer));
    l->node.type = ReLU2D;
    l->X = NULL;
    l->with_gradient = with_gradient;

    return l;
}

Activation2DLayer* CreateTanh2DLayer(int with_gradient) {
    Activation2DLayer* l = (Activation2DLayer*) malloc(sizeof(Activation2DLayer));
    l->node.type = Tanh2D;
    l->X = NULL;
    l->with_gradient = with_gradient;

    return l;
}

void DestroyActivation2DLayer(Activation2DLayer* layer) {
    if (layer->X != NULL) DestroyData2D(layer->X);
    free(layer);
}

LinearLayer* CreateLinearLayer(int in, int out, int with_gradient, int random) {
    LinearLayer* l = (LinearLayer*) malloc(sizeof(LinearLayer));
    l->w = fmatrix_allocate_2d(out, in);
    l->b = (float*) malloc(sizeof(float)*out);
    l->in = in;
    l->out = out;

    l->dW = with_gradient ? fmatrix_allocate_2d(out, in) : NULL;
    l->db = with_gradient ? (float*) malloc(sizeof(float)*out) : NULL;
    l->X = NULL;

    l->node.next = NULL;
    l->node.previous = NULL;
    l->node.type = Linear;

    if (random) {
        RandomInitLinearLayer(l);
    }

    return l;
}

void RandomInitLinearLayer(LinearLayer* l) {
    random_init_matrix(l->w, l->out, l->in);
    random_init_vector(l->b, l->out);
}

void LearnLinearLayer(LinearLayer* l, float learning_rate) {
    assert(l->dW != NULL && "Gradient was not calculated for this linear layer");

    #pragma omp parallel for
    for (int i = 0; i < l->out; i++) {
        //l->b[i] -= l->db[i] * learning_rate;
        for (int j = 0; j < l->in; j++) 
            l->w[i][j] -= l->dW[i][j] * learning_rate;
    }
}

float GetLinearLayerNorm(LinearLayer* l) {
    float sum = 0.0;
    for (int i = 0; i < l->out; i++)
        for (int j = 0; j < l->in; j++) 
            sum += fabsf(l->w[i][j]);
    return sum;
}

void DestroyLinearLayer(LinearLayer* layer) {
    free_fmatrix_2d(layer->w);
    free(layer->b);
    if (layer->dW != NULL) free_fmatrix_2d(layer->dW);
    if (layer->db != NULL) free(layer->db);
    if (layer->X != NULL) DestroyData1D(layer->X);
    layer->w = NULL;
    layer->dW = NULL;
    free(layer);
}

ConvLayer* CreateConvLayer(int in_channels, int out_channels, int size, int with_gradient, int random) {
    ConvLayer* c = (ConvLayer*) malloc(sizeof(ConvLayer));
    c->w = square_allocate_2d(out_channels, in_channels);
    c->b = (float*) malloc(sizeof(float)*out_channels);
    c->dW = with_gradient ? square_allocate_2d(out_channels, in_channels) : NULL;
    c->db = with_gradient ? (float*) malloc(sizeof(float)*out_channels) : NULL;
    c->X = NULL;

    for (int i=0; i<out_channels; i++) 
        for (int j=0; j<in_channels; j++) {
            c->w[i][j].mat = fmatrix_allocate_2d(size, size);
            c->w[i][j].size = size;
            if (with_gradient) {
                c->dW[i][j].mat = fmatrix_allocate_2d(size, size);
                c->dW[i][j].size = size;
            }
        }

    c->in = in_channels;
    c->out = out_channels;
    c->size = size;

    c->node.next = NULL;
    c->node.previous = NULL;
    c->node.type = Conv;

    if (random) {
        RandomInitConvLayer(c);
    }

    return c;
}

void RandomInitConvLayer(ConvLayer* c) {
    int size = c->size;

    random_init_vector(c->b, c->out);

    for (int i=0; i < c->out; i++) {
        for (int j=0; j < c->in; j++) {
            random_init_matrix(c->w[i][j].mat, size, size);
        }
    }
}

void LearnConvLayer(ConvLayer* c, float learning_rate) {
    assert(c->dW != NULL && "Gradient was not calculated for this conv layer");

    //#pragma omp parallel for
    for (int i=0; i<c->out; i++) {
        c->b[i] -= c->db[i] * learning_rate;
        for (int j=0; j<c->in; j++)
            for(int k=0; k<c->size; k++)
                for(int l=0; l<c->size; l++)
                    c->w[i][j].mat[k][l] -= c->dW[i][j].mat[k][l] * learning_rate;
    }
}

float GetConvLayerNorm(ConvLayer* c) {
    float sum = 0.0;
    for (int i = 0; i < c->out; i++)
        for (int j = 0; j < c->in; j++) 
            for(int k=0; k< c->size; k++)
                for(int l=0; l< c->size; l++)
                    sum += fabsf(c->w[i][j].mat[k][l]);
    return sum;
}

void DestroyConvLayer(ConvLayer* c) {
    for (int i=0; i<c->out; i++) 
        for (int j=0; j<c->in; j++) {
            DestroySquareMatrix(c->w[i][j]);
            if (c->dW != NULL) {
                DestroySquareMatrix(c->dW[i][j]);
            } 
        }
    free(c->w[0]);
    free(c->w);
    free(c->b);
    c->w = NULL;
    if (c->dW != NULL) {
        free(c->dW[0]);
        free(c->dW);
        c->dW = NULL;
    }
    if (c->db != NULL) free(c->db);
    if (c->X != NULL) DestroyData2D(c->X);

    free(c);
}

void print_conv_layer(ConvLayer* layer) {
    printf("ConvLayer of size [%d, %d] for kernels of size %d\n", layer->out, layer->in, layer->size);
}

// Returns Y size of a convulation of a W of size filter_size
// on an X of size X_size
int get_output_size(int input_size, int filter_size) {
    return input_size - filter_size + 1;
} 

int get_input_size(int output_size, int filter_size) {
    return output_size + filter_size - 1;
}
