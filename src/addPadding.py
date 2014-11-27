import gzip
import cPickle
import numpy
import theano
import theano.tensor as T

dataset = '../data/mnist.pkl.gz'

f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def addThatPad(ndArrayToPad):
	npndarray = numpy.ndarray((ndArrayToPad[0].shape[0], ndArrayToPad[0].shape[1] + 240), dtype=numpy.float32)
	exampleCount = 0
	for example in ndArrayToPad[0]:
		pixelCount = 0
		for pixel in example:
			npndarray[exampleCount][66+(pixelCount/28)*4+pixelCount] = pixel
			pixelCount = pixelCount + 1
		exampleCount = exampleCount + 1
	return npndarray