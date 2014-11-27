import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

class ConvolutionLayer(object):
    """"Convolutional Layer of a CNN"""

    def __init__(self, rng, input, input_shape, filter_shape):
        
        assert input_shape[1] == filter_shape[1]
        self.input = input

        #TODO: initialize according to Lecun
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        W_bound = numpy.sqrt(6.0 / (fan_in + fan_out))
        
        #weights on the filters
        self.conv_W = theano.shared(
            numpy.asarray(
                rng.uniform(low = -W_bound,
                            high = W_bound,
                            size = filter_shape),
                dtype = theano.config.floatX),
            borrow = True
        )
        
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        conv_out = conv.conv2d(
            input = input,
            image_shape = input_shape,
            filters = self.conv_W,
            filter_shape = filter_shape,
        )
  
        #amplitude of activation function tanh. Magic from lecun
        amp = 1.7195
        #slope of activation function tanh at 0. Magic from lecun
        slope = 2.0 / 3.0
        a = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = amp * T.tanh(slope * a)

        self.params = [self.conv_W, self.b]
