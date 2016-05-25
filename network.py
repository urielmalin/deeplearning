#!/usr/bin/python
import random
import logging
import math
import numpy

import cPickle as pickle

import theano.tensor as T
from theano import function, pp, config
import lasagne
import lasagne.layers as L

from elf_parser import ElfParser
from consts import *
from utils import *

iteration_number = 1

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
    l_in = L.InputLayer((None, SEQUENCE_LEN, 256))
    l_forward = L.RecurrentLayer(l_in, num_units=16)
    l_backward = L.RecurrentLayer(l_in, num_units=16, backwards=True)
    l_concat = L.ConcatLayer([l_forward, l_backward])
    l_out = L.DenseLayer(l_concat, num_units=2, nonlinearity=T.nnet.softmax)
    return l_out

def build_finding_func(network, ):
    input_var = T.tensor3()
    relevant_part = T.ivector()
    target_var = T.ivector("target_output")
    output = L.get_output(network, input_var)
    output = T.argmax(output, axis=1)
    return function([input_var],  [output] )

def build_test_func(network):
    input_var = T.tensor3()
    target_var = T.ivector("target_output")
    output = L.get_output(network, input_var)
    loss = lasagne.objectives.categorical_crossentropy(output, target_var)
    loss = loss.mean() * (iteration_number/iteration_number)
    params = L.get_all_params(network, trainable=True)
    test_acc = T.mean(T.eq(T.argmax(output, axis=1), target_var),
                          dtype=config.floatX)
    output = T.argmax(output, axis=1)
    f = function([input_var, target_var],  [loss, test_acc, output] )
    return f

def build_train_func(network, learning_rate=LEARNING_RATE, converge_rate=CONVERGE):
    input_var = T.tensor3()
    target_var = T.ivector("target_output")
    iteration_number = T.scalar() 
    output = L.get_output(network, input_var)
    loss = lasagne.objectives.categorical_crossentropy(output, target_var)
    loss = loss.mean() * (iteration_number/iteration_number)
    params = L.get_all_params(network, trainable=True)
    rate = learning_rate / ((1+iteration_number/converge_rate) ** POWER)
    updates = lasagne.updates.rmsprop(loss, params, rate)
    test_acc = T.mean(T.eq(T.argmax(output, axis=1), target_var),
                          dtype=config.floatX)
    output = T.argmax(output, axis=1)
    f = function([input_var, target_var, iteration_number],  [loss, test_acc, output], updates=updates)
    return f

def make_batch(code, funcs, start, batch_size=BATCH_SIZE):
    if len(code) < start + batch_size:
        batch_size = len(code) - start
    selected_bytes = code[start : start + batch_size]
    is_funcs = numpy.zeros(batch_size - (SEQUENCE_LEN - 1), dtype="int8")
    is_func_ends = numpy.zeros(batch_size - (SEQUENCE_LEN - 1), dtype="int8")
    start = start + SEQUENCE_BEFORE
    batch_size = batch_size - SEQUENCE_BEFORE - SEQUENCE_AFTER
    end = start + batch_size
    for func in funcs:
        offset = func.offset
        size = func.size
        if offset >= start and offset < end: 
            is_funcs[offset - start] = 1
        if offset + size - 1 >= start and offset + size + 1 < end:
            is_func_ends[offset + size - 1 - start] = 1
    return selected_bytes, is_funcs, is_func_ends

def prepare_input_mat(text, start, size=MINIBATCH_SIZE):
    input_mat = []
    for i in xrange(size):
        temp = text[start + i: start + i + SEQUENCE_LEN]
        temp = [makeVector(byte) for byte in temp]
        input_mat.append(temp)
    return input_mat

def prepare_finding_minibatch(text, start, size=MINIBATCH_SIZE):
    return prepare_input_mat(text, start, size)

def prepare_test_minibatch(text, func_start, func_end, start, size=MINIBATCH_SIZE):
    input_mat = prepare_input_mat(text, start)
    func_start = func_start[start : start + size]
    func_end = func_end[start : start + size]
    return input_mat, func_start, func_end

def find_functions_in_file(ep, start_func, end_func):
    text, _ = ep.get_code_and_funcs()
    batch_size = len(text)
    batch_results = [0] * (batch_size - (SEQUENCE_LEN - 1))
    batch_end_results = [0] * (batch_size - (SEQUENCE_LEN - 1))
    for i in xrange(0, len(text), 1):
        minibatch_bytes = prepare_finding_minibatch(text, i, MINIBATCH_SIZE)
        output = start_func(minibatch_start_bytes)
        end_output = end_func(minibatch_end_bytes)
        for j in xrange(MINIBATCH_SIZE):
            batch_results[i + j] = output[j]
            batch_end_results[i + j] = end_output[j]
    return batch_results, batch_end_results 

def do_test_batch(ep, start_func, end_func, batch_size=BATCH_SIZE):
    global iteration_number
    loss_sum = 0
    end_loss_sum = 0

    batch_results = [0] * (batch_size - (SEQUENCE_LEN - 1))
    batch_end_results = [0] * (batch_size - (SEQUENCE_LEN - 1))
    text, funcs = ep.get_code_and_funcs()
    if ep.get_code_len() > batch_size:
        start_index = random.randint(0, len(text) - batch_size - 1)
    else:
        start_index = 0
    batch_data = make_batch(text, funcs, start_index, batch_size)
    batch_bytes = batch_data[0]
    batch_is_funcs = batch_data[1]
    batch_is_end_funcs = batch_data[2]
    for i in xrange(0, batch_size - (SEQUENCE_LEN - 1), MINIBATCH_SIZE):
        minibatch_data = prepare_test_minibatch(batch_bytes, batch_is_funcs, batch_is_end_funcs, i, MINIBATCH_SIZE)
        minibatch_bytes, minibatch_start_func, minibatch_end_func = minibatch_data
        iteration_number += 1
        loss, acc, output = start_func(minibatch_bytes, minibatch_start_func)
        loss_sum += loss
        end_loss, end_acc, end_output = end_func(minibatch_bytes, minibatch_end_func)
        end_loss_sum += end_loss
        for j in xrange(MINIBATCH_SIZE):
            batch_results[i+j] = output[j] 
            batch_end_results[i+j] = end_output[j] 

    for i in xrange(batch_size - (SEQUENCE_LEN - 1)):
        if batch_results[i] == 1:
            logging.info("func: %d %s" % (i + start_index, batch_is_funcs[i]))
        if batch_end_results[i] == 1:
            logging.info("end func: %d %s" % (i + start_index, batch_is_end_funcs[i]))

    stats = calc_stats(batch_results, batch_is_funcs)
    end_stats = calc_stats(batch_end_results, batch_is_end_funcs)
    return stats, end_stats, loss_sum / batch_size, end_loss_sum / batch_size

def do_train_batch(ep, start_func, end_func, batch_size=BATCH_SIZE):
    global iteration_number
    loss_sum = 0
    end_loss_sum = 0

    batch_results = [0] * (batch_size - (SEQUENCE_LEN - 1))
    batch_end_results = [0] * (batch_size - (SEQUENCE_LEN -1))
    text, funcs = ep.get_code_and_funcs()
    if ep.get_code_len() > batch_size:
        start_index = random.randint(0, len(text) - batch_size - 1)
    else:
        start_index = 0
    batch_data = make_batch(text, funcs, start_index, batch_size)
    batch_bytes = batch_data[0]
    batch_is_funcs = batch_data[1]
    batch_is_end_funcs = batch_data[2]
    for i in xrange(0, batch_size - (SEQUENCE_LEN - 1), MINIBATCH_SIZE):
        minibatch_data = prepare_test_minibatch(batch_bytes, batch_is_funcs, batch_is_end_funcs, i, MINIBATCH_SIZE)
        minibatch_bytes, minibatch_start_func, minibatch_end_func= minibatch_data
        iteration_number += 1
        loss, acc, output = start_func(minibatch_bytes, minibatch_start_func, iteration_number)
        loss_sum += loss
        end_loss, end_acc, end_output = end_func(minibatch_bytes, minibatch_end_func, iteration_number)
        end_loss_sum += end_loss
        for j in xrange(MINIBATCH_SIZE):
            batch_results[i+j] = output[j] 
            batch_end_results[i+j] = end_output[j]

    for i in xrange(batch_size - (SEQUENCE_LEN - 1)):
        if batch_results[i] == 1:
            logging.info("func: %d %s" % (i + start_index, batch_is_funcs[i]))
        if batch_end_results[i] == 1:
            logging.info("end func: %d %s" % (i + start_index, batch_is_end_funcs[i]))

    stats = calc_stats(batch_results, batch_is_funcs)
    end_stats = calc_stats(batch_end_results, batch_is_end_funcs)
    return stats, end_stats, loss_sum / batch_size, end_loss_sum / batch_size

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


