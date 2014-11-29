import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import neighbours

class SubsampleLayer(object):
    """ Sub-sampling layer in a convolutional network """
    
    def __init__(self, rng, input, input_shape, pool_size):

        #initialize the weights 2.4 is magic from LeCun
        #multiply before to counteract the mean that is calculated later on(we want the sum)
        fan_in = numpy.prod(pool_size)
        W_bound = 4 * 2.4 / fan_in

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

        pool_sum = neighbours.images2neibs(
            ten4 = input,
            neib_shape = pool_size
        )
        pool_mean = pool_sum.mean(axis=-1)
        pool_out = pool_mean.reshape([input.shape[0], input.shape[1], input.shape[2]/2, input.shape[3]/2])

        #amplitude of activation function tanh. Magic from lecun
        amp = 1.7195
        #slope of activation function tanh at 0. Magic from lecun
        slope = 2.0 / 3.0
        #TODO need to elementwise multiply by shared weights W here
        #a = subsample_out.prod(self.W.dimshuffle('x',0,'x','x')) + self.b.dimshuffle('x', 0, 'x', 'x')
        a = subsample_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = amp * T.tanh(slope * a)
        
        #TODO add self.W when above is solved
        self.params = [self.b]
