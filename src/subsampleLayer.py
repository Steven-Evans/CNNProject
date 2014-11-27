import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample


class SubsampleLayer(object):
    """ Sub-sampling layer in a convolutional network """
    
    def __init__(self, rng, input, input_shape, pool_size):

        #initialize the weights
        #TODO do according to lecun, I just made this up
        #fan_in and fan_out are most likely wrong
        fan_in = numpy.prod(input_shape[1:])
        fan_out = input_shape[0] * numpy.prod(input_shape[2:]) / 2
        W_bound = numpy.sqrt(6.0 / (fan_in + fan_out))

        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low = -W_bound,
                            high = W_bound,
                            size = input_shape[1]),
                dtype = theano.config.floatX),
            borrow = True
        )
        

        #the bias weights
        b_values = numpy.zeros((input_shape[1],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        subsample_out = downsample.max_pool_2d(
            input = input,
            ds = pool_size,
            ignore_border = True
        )

        #amplitude of activation function tanh. Magic from lecun
        amp = 1.7195
        #slope of activation function tanh at 0. Magic from lecun
        slope = 2.0 / 3.0
        #TODO need to elementwise multiply by shared weights W here
        a = (subsample_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = amp * T.tanh(slope * a)
        
        #TODO add self.W when above is solved
        self.params = [self.b]
