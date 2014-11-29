import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

class CustomConvLayer(object):
    """"Custom Convolutional Layer C1 in LeNet"""

    def __init__(self, rng, input, input_shape, filter_shape):
        
        assert input_shape[1] == filter_shape[1]
        self.input = input
        
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        convs, self.W = self.buildConnections(rng, input, input_shape, filter_shape)
        
        #concatenate using T.concatenate([x0, x1, ..., x15], axis=1) where
        #  each x is of shape(batch_size, 1, 12, 12)        
        conv_out = T.concatenate(convs, axis=1)
        
        #amplitude of activation function tanh. Magic from lecun
        amp = 1.7195
        #slope of activation function tanh at 0. Magic from lecun
        slope = 2.0 / 3.0
        a = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = amp * T.tanh(slope * a)
        
        self.params = self.W + [self.b]

    def buildConnections(self, rng, input, input_shape, filter_shape):
        
        # Table of feature map connectivity
        # Row represents input feature map and cols represent output feature maps
        #
        #   0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
        # 0 x       x x x     x  x  x  x     x  x
        # 1 x x       x x x      x  x  x  x     x
        # 2 x x x       x x x       x     x  x  x
        # 3   x x x     x x x x        x     x  x
        # 4     x x x     x x x  x     x  x     x
        # 5       x x x     x x  x  x     x  x  x
        
        #implementation of above table
        connections = [(0, 1, 2),
                (1, 2, 3),
                (2, 3, 4),
                (3, 4, 5),
                (4, 5, 0),
                (5, 0, 1),
                (0, 1, 2, 3),
                (1, 2, 3, 4),
                (2, 3, 4, 5),
                (3, 4, 5, 0),
                (4, 5, 0, 1),
                (5, 0, 1, 2),
                (0, 1, 3, 4),
                (1, 2, 4, 5),
                (2, 3, 5, 0),
                (0, 1, 2, 3, 4, 5)]

        convs = []
        W = []

        #fan_in = numpy.prod(filter_shape[1:])
        #fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        #W_bound = numpy.sqrt(6.0 / (fan_in + fan_out))

        for i in xrange(len(connections)):
            fmap_shape = (1, len(connections[i]), filter_shape[2], filter_shape[3])
            #note: this should also be multiplied by len(connections[i]) but wont converge
            fan_in = numpy.prod(filter_shape[2:])
            #2.4 is magic from LeCun
            W_bound = 2.4 / fan_in

            W.append(self.generateWeights(rng, fmap_shape, W_bound))
            convs.append(conv.conv2d(
                input = input[:,connections[i],:,:],
                image_shape = (input_shape[0], len(connections[i]), input_shape[2], input_shape[3]),
                filters = W[i],
                filter_shape = fmap_shape
            ))

        return [convs, W]

    def generateWeights(self, rng, size, W_bound):
        return theano.shared(
            numpy.asarray(
                rng.uniform(low = -W_bound,
                            high = W_bound,
                            size = size),
                dtype = theano.config.floatX),
            borrow = True
        )
