import os
import random
import sys
import numpy
import theano.tensor as T
from theano import function, pp, config
import lasagne
import lasagne.layers as L
from elf_parser import ElfParser
from consts import *

def makeMatrix(next_bytes):
    mat = numpy.zeros((len(next_bytes),256),dtype="int8")
    for i in xrange(len(next_bytes)):
        mat[i][next_bytes[i]] = True
    return mat

def makeVector(byte):
   l = numpy.zeros((256), dtype="int8")
   l[ord(byte)] = True 
   return l

def build_network(input_var=None):
    l_in = L.InputLayer((None,1, 256),input_var=input_var)
    l_forward = L.RecurrentLayer(l_in, num_units=16)
    l_backward = L.RecurrentLayer(l_in, num_units=16, backwards=True)
    l_concat = L.ConcatLayer([l_forward, l_backward])
    l_out = L.DenseLayer(l_concat, num_units=2, nonlinearity=T.nnet.softmax)
    return l_out

def build_train_func(input_var, network):
    target_var = T.ivector("target_output")
    output = L.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(output, target_var)
    loss = loss.mean()
    params = L.get_all_params(network)
    updates = lasagne.updates.rmsprop(loss, params, LEARNING_RATE)
    test_acc = T.mean(T.eq(T.argmax(output, axis=1), target_var),
                          dtype=config.floatX)
    output = T.argmax(output, axis=1)
    f = function([input_var, target_var],  [loss, test_acc, output], updates=updates)
    return f

def make_batch(code, funcs, start, batch_size=BATCH_SIZE):
    selected_bytes = code[start : start + batch_size]
    is_funcs = numpy.zeros(batch_size, dtype="int8")
    for offset in funcs.get_offsets_list():
        if offset >= start and offset <= start + batch_size:
            is_funcs[offset - start] = 1
    return selected_bytes, is_funcs

input_var = T.tensor3()
print "building network..."
network = build_network(input_var)
print "building train function..."
train = build_train_func(input_var, network)

def do_batch(ep, batch_size=BATCH_SIZE):
    text, funcs = ep.get_code_and_funcs()
    start_index = random.randint(0, len(text) - batch_size - 1)
    batch_bytes, batch_is_funcs = make_batch(text, funcs, start_index, batch_size)
    for i in xrange(0, batch_size, MINIBATCH_SIZE):
        minibatch_bytes = batch_bytes[i : i + MINIBATCH_SIZE]
        minibatch_bytes = [[makeVector(byte)] for byte in minibatch_bytes]
        minibatch_is_funcs = batch_is_funcs[i : i + MINIBATCH_SIZE]
        loss, acc, output = train(minibatch_bytes, minibatch_is_funcs)
        for j in xrange(len(minibatch_bytes)):
            if output[j] == 1:
                print "func: ", i+j, funcs.is_offset_start(i+j)
        print loss, acc 


def main(argv):
    print "start training"
    files = os.listdir(argv[0])
    l = []
    for f in files:
        l.append(ElfParser(os.path.join(argv[0], f), f))
    for i in xrange(3):
        index = random.randint(0, len(l) - 1)
        print "file: " , f[index][1]
        do_batch(l[index][0])
     

    


    


if __name__ == "__main__":
    main(sys.argv)
