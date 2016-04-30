#!/usr/bin/python
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

def makeMatrix(next_bytes):
    mat = numpy.zeros((len(next_bytes),256),dtype="int8")
    for i in xrange(len(next_bytes)):
        mat[i][next_bytes[i]] = True
    return mat

def makeVector(byte):
   l = numpy.zeros((256), dtype="int8")
   l[ord(byte)] = True 
   return l

def build_network():
    l_in = L.InputLayer((None,1, 256))
    l_forward = L.RecurrentLayer(l_in, num_units=16)
    l_backward = L.RecurrentLayer(l_in, num_units=16, backwards=True)
    l_concat = L.ConcatLayer([l_forward, l_backward])
    l_out = L.DenseLayer(l_concat, num_units=2, nonlinearity=T.nnet.softmax)
    return l_out

def build_func(network, is_train):
    input_var = T.tensor3()
    target_var = T.ivector("target_output")
    output = L.get_output(network, input_var)
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

def calc_precision(stats):
    if stats[0] + stats[1] == 0:
        return 0
    return float(stats[0]) / (stats[0] + stats[1])

def calc_recall(stats):
    if stats[0] + stats[2] == 0:
        return 0
    return float(stats[0]) / (stats[0] + stats[2])

def calc_F1(stats):
    precision = calc_precision(stats)
    recall = calc_precision(stats)
    if precision == 0 or recall == 0:
        return 0
    return (2 * precision * recall) / (precision / recall)

def calc_stats(output, target):
    if len(output) != len(target):
        raise Exception ("Lists length isnt equal")
    TP = 0
    FN = 0
    FP = 0
    for predicated_byte, target_byte in zip(output, target):
        if predicated_byte == target_byte and target_byte == True:
            TP += 1
        elif predicated_byte > target_byte:
            FP += 1
        elif target_byte > predicated_byte:
            FN += 1
    return TP, FP, FN


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
        if offset + size - 1>= start and offset + size + 1 < start + batch_size:
            is_func_ends[offset + size - 1 - start] = 1
    return selected_bytes, is_funcs, is_func_ends

def do_batch(ep, start_func, end_func, reverse_start=True, batch_size=BATCH_SIZE):
    batch_results = [0] * batch_size
    batch_end_results = [0] * batch_size
    text, funcs = ep.get_code_and_funcs()
    if ep.get_code_len() > batch_size:
        start_index = random.randint(0, len(text) - batch_size - 1)
    else:
        start_index = 0
    batch_data = make_batch(text, funcs, start_index, batch_size)
    batch_bytes = batch_data[0]
    batch_is_funcs = batch_data[1]
    batch_is_end_funcs = batch_data[2]
    for i in xrange(0, batch_size, 1):
        minibatch_bytes = batch_bytes[i : i + MINIBATCH_SIZE]
        minibatch_bytes = [[makeVector(byte)] for byte in minibatch_bytes]
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
                batch_results[i + j] = 1
            if end_output[j] == 1:
                batch_end_results[i + j] = 1
        logging.debug("start: %lf, %lf | end: %lf, %lf" % (loss, acc, end_loss, end_acc))
    for i in xrange(batch_size):
        if batch_results[i] == 1:
            logging.info("func: %d %s" % (i + start_index, batch_is_funcs[i]))
        if batch_end_results[i] == 1:
            logging.info("end func: %d %s" % (i + start_index, batch_is_end_funcs[i]))

    stats = calc_stats(batch_results, batch_is_funcs)
    end_stats = calc_stats(batch_end_results, batch_is_end_funcs)
    return stats, end_stats

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


