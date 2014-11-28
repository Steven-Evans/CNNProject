import os
import gzip
import cPickle
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from addPadding import addThatPad

#adapted from http://deeplearning.net/tutorial/logreg.html

def load_MNIST():
    normalized_file = '../data/normalized-mnist.dmp'
    if os.path.isfile(normalized_file):
        with open(normalized_file, 'rb') as f:
            return cPickle.load(f)


    dataset_file = '../data/mnist.pkl.gz'
    dataset_url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    #download the dataset if it isn't in the current folder
    if not os.path.isfile(dataset_file):
        import urllib
        origin = (dataset_url)
        print 'Downloading MNIST from %s' % dataset_url
        print '...downloading'
        urllib.urlretrieve(origin, dataset_file)
    
    print '...loading data'

    #load the dataset
    f = gzip.open(dataset_file, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    #don't want a validation set, add to training set
#    train_set = (numpy.concatenate((train_set[0], valid_set[0])), 
#                 numpy.concatenate((train_set[1], valid_set[1])))

    #pad images so that they're 32x32 instead of 28x28
    #TODO add the padding

    test_set = addThatPad(test_set)
    train_set = addThatPad(train_set)
    valid_set = addThatPad(valid_set)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
        
    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)

    rval = [(train_set_x, train_set_y),  (valid_set_x, valid_set_y), \
            (test_set_x, test_set_y)]

    #write the normalize file so we don't have to normalize it again
    with open(normalized_file,'wb') as f:
        cPickle.dump(rval, f, -1)

    return rval

if __name__ == '__main__':
    datasets = load_MNIST()

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    #example how to recover the data
    print(train_set_x.get_value().shape)
    print(train_set_y.owner.inputs[0].get_value().shape)
    print(test_set_x.get_value().shape)
    print(test_set_y.owner.inputs[0].get_value().shape)

    #example printing out a digit
    plt.gray()
    plt.imshow(1 - train_set_x.get_value()[0].reshape(32,32))
    plt.show()
        
