#!/usr/bin/python
import logging

import pylab

def add_lists(a, b):
    return map(lambda x, y: x+y, a, b)

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
    recall = calc_recall(stats)
    if precision + recall == 0:
        return 0
    return (2 * precision * recall) / float(precision + recall)

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

def build_short_stats(stats):
   s = "%d TP, %d FP, %d FN" % (stats[0], stats[1], stats[2])
   s += " | P: %lf, R: %lf, F1: %lf" % (calc_precision(stats), calc_recall(stats), calc_F1(stats))
   return s

def print_batch_stats(stats, end_stats, loss, end_loss):
    logging.info("=== Start: %s | loss: %lf ===" % (build_short_stats(stats), loss))
    logging.info("=== End:   %s | loss: %lf ===" % (build_short_stats(end_stats), end_loss))

def print_stats(stats, end_stats):
    logging.info("start: %d TP, %d FP, %d FN" % (stats[0], stats[1], stats[2]))
    logging.info("Precision: %lf, Recall: %lf, F1: %lf" % (calc_precision(stats), calc_recall(stats), calc_F1(stats)))
    logging.info("end: %d TP, %d FP, %d FN" % (end_stats[0], end_stats[1], end_stats[2]))
    logging.info("Precision: %lf, Recall: %lf, F1: %lf" % (calc_precision(end_stats), calc_recall(end_stats), calc_F1(end_stats)))


def draw_plot(x_label, y_label, start_values, end_values, filename,figure):
    x = range(1, len(start_values) + 1)   
    pylab.figure(figure)
    pylab.plot(x, start_values, "-b", label="start")
    pylab.plot(x, end_values, "-r", label="end")
    pylab.legend(loc="upper right")
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.savefig(filename)
    # pylab.show()
