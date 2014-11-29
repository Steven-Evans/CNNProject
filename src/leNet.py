import time

import numpy

import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from loadMNIST import load_MNIST
from convPoolLayer import ConvPoolLayer
from hiddenLayer import HiddenLayer
from classifiers import LogisticRegression
from convolutionLayer import ConvolutionLayer
from subsampleLayer import SubsampleLayer
from customConvLayer import CustomConvLayer
from webpageDisplay import createWebpage

def build_model(datasets, batch_size, rng, learning_rate):

    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    #reshape the image as input to the first conv pool layer
    #MNIST images are 28x28
    layer0_input = x.reshape((batch_size, 1, 32, 32))
    
    layer0_conv = ConvolutionLayer(
        rng = rng,
        input = layer0_input,
        input_shape = (batch_size, 1, 32, 32),
        filter_shape = (6, 1, 5, 5)
    )

    layer0_subsample = SubsampleLayer(
        rng = rng,
        input = layer0_conv.output,
        input_shape = (batch_size, 6, 28, 28),
        pool_size = (2, 2)
    )

    #the custom convolution layer: C4 in lecun
    layer1_conv = CustomConvLayer(
        rng = rng,
        input = layer0_subsample.output,
        input_shape = (batch_size, 6, 14, 14),
        filter_shape = (16, 6, 5, 5)
    )
    
    layer1_subsample = SubsampleLayer(
        rng = rng,
        input = layer1_conv.output,
        input_shape = (batch_size, 16, 10, 10),
        pool_size = (2, 2)
    )

    layer2_conv = ConvolutionLayer(
        rng = rng,
        input = layer1_subsample.output,
        input_shape = (batch_size, 16, 5, 5),
        filter_shape = (120, 16, 5, 5)
    )

    #flatten the output of the convpool layer for input to the MLP layer
    layer3_input = layer2_conv.output.flatten(2)

    layer3 = HiddenLayer(
        rng,
        input = layer3_input,
        n_in = 120,
        n_out = 84,
        activation = T.tanh
    )
    
    #TODO: Change to RBF
    layer4 = LogisticRegression(
        input = layer3.output,
        n_in = 84,
        n_out = 10
    )
    
    cost = layer4.negative_log_likelihood(y)
    
    params = layer4.params + layer3.params + \
             layer2_conv.params + \
             layer1_conv.params + layer1_subsample.params + \
             layer0_conv.params + layer0_subsample.params

    gradients = T.grad(cost, params)
    
    updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, gradients)]

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    train_model = theano.function(
        [index],
        cost,
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens = {
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_pred = theano.function(
        [index],
        layer4.y_pred,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
        }
    )

    test_pred = theano.function(
        [index],
        layer4.y_pred,
        givens = {
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
        }
    )

    return (train_model, train_pred, test_model, test_pred)

def train_LeNet(datasets):
    learning_rate = theano.shared(numpy.cast[theano.config.floatX](0),
                                  borrow = True)

    learning_rates = [0.001] \
                     + [0.0005]*3 \
                     + [.0002]*3 \
                     + [.0001]*4 \
                     + [.00005]*4 \
                     + [.00001]*5

    n_train = datasets[0][0].get_value(borrow=True).shape[0]
    n_test = datasets[1][0].get_value(borrow=True).shape[0]

    rng = numpy.random.RandomState(23455)

    print('building model...')
    train_model, train_pred, test_model, test_pred = build_model(datasets, 1, rng, learning_rate)

    train_score = []
    test_score = []

    print('training...')
    start_time = time.clock()

    for iter in xrange(0,len(learning_rates)):
        learning_rate.set_value(learning_rates[iter])
        
        train_losses = [train_model(i) for i in xrange(n_train)]
        train_score.append(numpy.mean(train_losses))

        test_losses = [test_model(i) for i in xrange(n_test)]
        test_score.append(numpy.mean(test_losses))
        print(('iter %i, test error: %f %%') %
              (iter, test_score[iter] * 100.0))

    end_time = time.clock()

    test_preds = [test_pred(i)[0] for i in xrange(n_test)]
    createWebpage(datasets, test_preds)
    
    print(('Optimization completed %i iterations in %.2f mins '
          'with final error rate of %f%%') %
          (len(learning_rates), ((end_time - start_time) / 60.0),
           test_score[len(learning_rates)-1] * 100.0))

    printGraph(train_score, test_score)

def printGraph(train_score, test_score):
    train_score = [x * 100.0 for x in train_score]
    test_score = [x * 100.0 for x in test_score]

    plt.plot(train_score, 'bs')
    plt.plot(test_score, 'g^')

    plt.ylabel('Error Rate %')
    plt.xlabel('Iteration #')
    plt.title('Error Rate of Training a CNN')
    plt.legend(['Train Error', 'Test Error'])
    
    plt.show()

if __name__ == '__main__':
    datasets = load_MNIST()
    train_LeNet(datasets)
     
