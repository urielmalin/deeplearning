<<<<<<< HEAD
import random
import logging
import numpy

import cPickle as pickle

import theano.tensor as T
from theano import function, pp, config
import lasagne
import lasagne.layers as L

from elf_parser import ElfParser
from consts import *
from utils import *
=======
import os
import time
import random
import sys
import argparse
import logging
import numpy
import theano.tensor as T
import cPickle as pickle

from theano import function, pp, config
import lasagne
import lasagne.layers as L
from elf_parser import ElfParser
from consts import *
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c

def makeMatrix(next_bytes):
    mat = numpy.zeros((len(next_bytes),256),dtype="int8")
    for i in xrange(len(next_bytes)):
        mat[i][next_bytes[i]] = True
    return mat

def makeVector(byte):
   l = numpy.zeros((256), dtype="int8")
   l[ord(byte)] = True 
   return l

<<<<<<< HEAD
def build_network():
    l_in = L.InputLayer((None,1, 256))
=======
def build_network(input_var=None):
    l_in = L.InputLayer((None,1, 256),input_var=input_var)
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c
    l_forward = L.RecurrentLayer(l_in, num_units=16)
    l_backward = L.RecurrentLayer(l_in, num_units=16, backwards=True)
    l_concat = L.ConcatLayer([l_forward, l_backward])
    l_out = L.DenseLayer(l_concat, num_units=2, nonlinearity=T.nnet.softmax)
    return l_out

def build_func(network, is_train):
    input_var = T.tensor3()
    target_var = T.ivector("target_output")
<<<<<<< HEAD
    output = L.get_output(network, input_var)
=======
    output = L.get_output(network)
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c
    loss = lasagne.objectives.categorical_crossentropy(output, target_var)
    loss = loss.mean()
    params = L.get_all_params(network)
    updates = lasagne.updates.rmsprop(loss, params, LEARNING_RATE)
    test_acc = T.mean(T.eq(T.argmax(output, axis=1), target_var),
                          dtype=config.floatX)
    output = T.argmax(output, axis=1)
    if is_train:
        f = function([input_var, target_var],  [loss, test_acc, output], updates=updates)
    else:
        f = function([input_var, target_var],  [loss, test_acc, output])
    return f

<<<<<<< HEAD
def calc_precision(stats):
    if stats[0] + stats[1] == 0:
        return 0
    return float(stats[0]) / (stats[0] + stats[1])

def calc_recall(stats):
    if stats[0] + stats[1] == 0:
        return 0
    return float(stats[0]) / (stats[0] + stats[1])

def calc_F1(stats):
    precision = calc_precision(stats)
    recall = calc_precision(stats)
    if precision == 0 or recall == 0:
        return 0
    return (2 * precision * recall) / (precision / recall)

def calc_stats(output, target):
=======
def calc_precision(TP, FP):
    return float(TP) / (TP + FP)

def calc_recall(TP, FN):
    return float(TP) / (TP + FN)

def calc_F1(precision, recall):
    return (2 * precision * recall) / (precision / recall)

def calc_score(output, target):
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c
    if len(output) != len(target):
        raise Exception ("Lists length isnt equal")
    TP = 0
    FN = 0
    FP = 0
<<<<<<< HEAD
    for predicated_byte, target_byte in zip(output, target):
=======
    for predicated_byte, traget_byte in zip(output, target):
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c
        if predicated_byte == target_byte and target_byte == True:
            TP += 1
        elif predicated_byte > target_byte:
            FP += 1
        elif target_byte > predicated_byte:
            FN += 1
<<<<<<< HEAD
    return TP, FP, FN
=======
    precision = calc_precision(TP, FP)
    recall = calc_recall(TP, NF)
    F1 = calc_F1(precision, recall) 
    return F1, precision, recall, TP, FP, FN
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c


def make_batch(code, funcs, start, batch_size=BATCH_SIZE):
    if len(code) < start + batch_size:
        batch_size = len(code) - start
    selected_bytes = code[start : start + batch_size]
    is_funcs = numpy.zeros(batch_size, dtype="int8")
    is_func_ends = numpy.zeros(batch_size, dtype="int8")
    for func in funcs:
        offset = func.offset
        size = func.size
        if offset >= start and offset < start + batch_size:
            is_funcs[offset - start] = 1
        elif offset + size - 1>= start and offset + size + 1 < start + batch_size:
<<<<<<< HEAD
            is_func_ends[offset + size - 1 - start] = 1
    return selected_bytes, is_funcs, is_func_ends

def do_batch(ep, start_func, end_func, reverse_start=True, batch_size=BATCH_SIZE):
    stats = [0, 0, 0]
    end_stats = [0, 0, 0]
    text, funcs = ep.get_code_and_funcs()
    if ep.get_code_len() > batch_size:
        start_index = random.randint(0, len(text) - batch_size - 1)
    else:
        start_index = 0
=======
            is_func_ends[offset + size -1 + start] = 1
    return selected_bytes, is_funcs, is_func_ends

def init_network():
    input_var = T.tensor3()
    print "building network..."
    network = build_network(input_var)
    return network

def do_batch(ep, start_func, end_func, batch_size=BATCH_SIZE):
    text, funcs = ep.get_code_and_funcs()
    start_index = random.randint(0, len(text) - batch_size - 1)
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c
    batch_data = make_batch(text, funcs, start_index, batch_size)
    batch_bytes = batch_data[0]
    batch_is_funcs = batch_data[1]
    batch_is_end_funcs = batch_data[2]
    for i in xrange(0, batch_size, MINIBATCH_SIZE):
        minibatch_bytes = batch_bytes[i : i + MINIBATCH_SIZE]
        minibatch_bytes = [[makeVector(byte)] for byte in minibatch_bytes]
<<<<<<< HEAD
        minibatch_is_end_funcs = batch_is_end_funcs[i : i + MINIBATCH_SIZE]
        end_loss, end_acc, end_output = end_func(minibatch_bytes, minibatch_is_end_funcs)
        minibatch_is_funcs = batch_is_funcs[i : i + MINIBATCH_SIZE]
        if reverse_start:
            loss, acc, output = start_func(minibatch_bytes[::-1], minibatch_is_funcs[::-1])
            output = output[::-1]
        else:
            loss, acc, output = start_func(minibatch_bytes, minibatch_is_funcs)
        for j in xrange(len(minibatch_bytes)):
            if output[j] == 1:
                logging.info("func: %d %s" % (i + j + start_index, minibatch_is_funcs[j]))
            if end_output[j] == 1:
                logging.info("end func: %d %s" % (i + j + start_index, minibatch_is_end_funcs[j]))
        stats = add_lists(stats, calc_stats(output, minibatch_is_funcs))
        end_stats = add_lists(end_stats, calc_stats(end_output, minibatch_is_end_funcs))
        logging.debug("start: %lf, %lf | end: %lf, %lf" % (loss, acc, end_loss, end_acc))
    return stats, end_stats
=======
        minibatch_is_funcs = batch_is_funcs[i : i + MINIBATCH_SIZE][::-1]
        minibatch_is_end_funcs = batch_is_end_funcs[i : i + MINIBATCH_SIZE]
        loss, acc, output = start_func(minibatch_bytes[::-1], minibatch_is_funcs)
        end_loss, end_acc, end_output = end_func(minibatch_bytes, minibatch_is_end_funcs)
        for j in xrange(len(minibatch_bytes)):
            if output[j] == 1:
                print "func: ", i+(MINIBATCH_SIZE - j+1)+start_index, minibatch_is_funcs[j]
            if end_output[j] == 1:
                print "end func: ", i + j + start_index, minibatch_is_end_funcs[j]
        print "start: %lf, %lf | end: %lf, %lf" % (loss, acc, end_loss, end_acc)
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c

def save_model(start_network, end_network, filename):
    data = []
    data.append(L.get_all_param_values(start_network))
    data.append(L.get_all_param_values(end_network))
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_model(start_network, end_network, filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    L.set_all_param_values(start_network, data[0])
    L.set_all_param_values(end_network, data[1])


<<<<<<< HEAD
=======
def main(argv):

    parser = argparse.ArgumentParser(description='Binary functions recognization neural network')

    parser.add_argument('-d', "--data-dir", action="store", dest="data_dir", required=True)
    parser.add_argument('--train', action="store_true", dest="is_train", default=True, help="Train the network")
    parser.add_argument('--test', action="store_false", dest="is_train")
    parser.add_argument('-m', "--load-model", action="store", dest="load_model")
    parser.add_argument('-s', "--save-model", action="store", dest="save_model")
    parser.add_argument('-l', "--log", action="store", dest="log_file")
    parser.add_argument('-t', "--time", action="store", dest="train_time", type=int, default=60*60*2)

    args = parser.parse_args(argv)

    start_network = init_network()
    end_network = init_network()
    if args.load_model != None:
        load_model(start_network, end_network, args.load_model)
    network_start_func = build_func(start_network, args.is_train)
    network_end_func = build_func(end_network, args.is_train)
    files = os.listdir(args.data_dir)
    l = []
    for f in files:
        l.append([ElfParser(os.path.join(args.data_dir, f)), f])
    sum_f = 0 
    sum_b = 0
    for ep in l:
        text, funcs = ep[0].get_code_and_funcs()
        print "%s: %d funcs, %d bytes" % (ep[1], len(funcs), len(text))
        sum_f += len(funcs)
        sum_b += len(text)

    print "Sum of funcs: %d, bytes: %d", (sum_f, sum_b)
    start = time.time()
    i = 0
    while time.time() - start <= args.train_time:
        index = random.randint(0, len(l) - 1)
        print "file: " , l[index][1]
        do_batch(l[index][0])
        i += 1
    print "Did %d batches with %d bytes" % (i, i * BATCH_SIZE)
    if args.save_model != None
        save_model(start_network, end_network, args.save_model)
     

    


    


if __name__ == "__main__":
    main(sys.argv)
>>>>>>> be9a4d6875e2b4949b8ab3e1e5c388a4edcb963c
