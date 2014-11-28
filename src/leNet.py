import time

import numpy

import theano
import theano.tensor as T

from loadMNIST import load_MNIST
from convPoolLayer import ConvPoolLayer
from hiddenLayer import HiddenLayer
from classifiers import LogisticRegression
from convolutionLayer import ConvolutionLayer
from subsampleLayer import SubsampleLayer
from customConvLayer import CustomConvLayer


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
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_model = theano.function(
        [index],
        cost,
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    valid_model = theano.function(
        [index],
        layer4.errors(y),
        givens = {
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
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

    return (train_model, valid_model, test_model)
    
    
def train_LeNet(datasets):
    
    batch_size = 500
    learning_rate = 0.1
    rng = numpy.random.RandomState(23455)
    
    train_model, valid_model, test_model = build_model(datasets, batch_size, rng, learning_rate)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = datasets[0][0].get_value(borrow=True).shape[0]
    n_valid_batches = datasets[1][0].get_value(borrow=True).shape[0]
    n_test_batches = datasets[2][0].get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    
    n_epochs = 200
    patience = 10000
    patience_increase = 2

    improvement_threshold = 0.995
    
    validation_frequency = min(n_train_batches, patience /2)

    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if iter % 100 == 0:
                print 'training @ iter = ', iter
            
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                
                validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.0))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    #if checking test error
                    #test_losses = [ test_model(i) for i in xrange(n_test_batches)]
                    #test_score = numpy.mean(test_losses)
                    #print(('    epoch %i, minibatch %i/%i, test error of best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches, test_score * 100.0))
            if patience <= iter:
                print('made it')
                done_looping = True
                break
                
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i' % 
          (best_validation_loss * 100.0, best_iter + 1))

def test_LeNet():
    datasets = load_MNIST()
    train_LeNet(datasets)


if __name__ == '__main__':
    test_LeNet()
     
