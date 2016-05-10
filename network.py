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
    l_in = L.InputLayer((None,1, 256))
    l_forward = L.RecurrentLayer(l_in, num_units=16)
    l_backward = L.RecurrentLayer(l_in, num_units=16, backwards=True)
    l_concat = L.ConcatLayer([l_forward, l_backward])
    l_out = L.DenseLayer(l_concat, num_units=2, nonlinearity=T.nnet.softmax)
    return l_out

def build_func(network, is_train, learning_rate=LEARNING_RATE, converge_rate=CONVERGE):
    input_var = T.tensor3()
    relevant_part = T.ivector()
    target_var = T.ivector("target_output")
    iteration_number = T.scalar() 
    output = L.get_output(network, input_var)
    output1 = output[relevant_part[0] : relevant_part[1]]
    target_var1 = target_var[relevant_part[0] : relevant_part[1]]
    loss = lasagne.objectives.categorical_crossentropy(output1, target_var1)
    loss = loss.mean() * (iteration_number/iteration_number)
    params = L.get_all_params(network, trainable=True)
    rate = learning_rate / ((1+iteration_number/converge_rate) ** 0.5)
    updates = lasagne.updates.rmsprop(loss, params, rate)
    test_acc = T.mean(T.eq(T.argmax(output1, axis=1), target_var1),
                          dtype=config.floatX)
    output1 = T.argmax(output1, axis=1)
    if is_train:
        f = function([input_var, target_var, iteration_number, relevant_part],  [loss, test_acc, output1], updates=updates)
    else:
        f = function([input_var, target_var, iteration_number, relevant_part],  [loss, test_acc, output1] )
    return f

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

def do_batch(ep, start_func, end_func, reverse_start=False, reverse_end=True, batch_size=BATCH_SIZE):
    global iteration_number
    loss_sum = 0
    end_loss_sum = 0

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
        iteration_number += 1
        padding_len = 0
        minibatch_bytes = batch_bytes[i : i + MINIBATCH_SIZE]
        minibatch_is_funcs = batch_is_funcs[i : i + MINIBATCH_SIZE]
        minibatch_is_end_funcs = batch_is_end_funcs[i : i + MINIBATCH_SIZE]
        if i + MINIBATCH_SIZE > batch_size:
            padding_len = i + MINIBATCH_SIZE - batch_size
            minibatch_bytes += "\x00" * padding_len
            padding_arr = numpy.zeros(padding_len, dtype="int8")
            minibatch_is_funcs = numpy.append(minibatch_is_funcs, padding_arr)
            minibatch_is_end_funcs = numpy.append(minibatch_is_end_funcs,padding_arr)
        minibatch_bytes = [[makeVector(byte)] for byte in minibatch_bytes]
        if reverse_end:
            relevant_part = [padding_len, MINIBATCH_SIZE]
            end_loss, end_acc, end_output = end_func(minibatch_bytes[::-1], minibatch_is_end_funcs[::-1], iteration_number, relevant_part)
            end_output = end_output[::-1]
        else:
            relevant_part = [0, MINIBATCH_SIZE - padding_len]
            end_loss, end_acc, end_output = end_func(minibatch_bytes, minibatch_is_end_funcs, iteration_number, relevant_part)
        end_loss_sum += end_loss
        if reverse_start:
            relevant_part = [padding_len, MINIBATCH_SIZE]
            loss, acc, output = start_func(minibatch_bytes[::-1], minibatch_is_funcs[::-1], iteration_number, relevant_part)
            output = output[::-1]

        else:
            relevant_part = [0, MINIBATCH_SIZE - padding_len]
            loss, acc, output = start_func(minibatch_bytes, minibatch_is_funcs, iteration_number, relevant_part)
        loss_sum += loss
        batch_results[i] = output[0] 
        batch_end_results[i] = end_output[0] 
        logging.debug("start: %lf, %lf | end: %lf, %lf" % (loss, acc, end_loss, end_acc))
    for i in xrange(batch_size):
        if batch_results[i] == 1:
            logging.info("func: %d %s" % (i + start_index, batch_is_funcs[i]))
        if batch_end_results[i] == 1:
            logging.info("end func: %d %s" % (i + start_index, batch_is_end_funcs[i]))

    stats = calc_stats(batch_results, batch_is_funcs)
    end_stats = calc_stats(batch_end_results, batch_is_end_funcs)
    return stats, end_stats, loss / batch_size, loss_sum / batch_size

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


