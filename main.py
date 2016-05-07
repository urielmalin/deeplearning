#!/usr/bin/python

from network import *
import os
import time
import sys
import argparse

def train(files_list, network_start_func, network_end_func, reverse_start, reverse_end, trainning_time):
    sum_f = 0 
    sum_b = 0
    for ep in files_list:
        text, funcs = ep[0].get_code_and_funcs()
        logging.info("%s: %d funcs, %d bytes" % (ep[1], len(funcs), len(text)))
        sum_f += len(funcs)
        sum_b += len(text)

    logging.info("Sum of funcs: %d, bytes: %d" % (sum_f, sum_b))

    stats = [0, 0, 0]
    end_stats = [0, 0, 0]
    start = time.time()
    i = 0
    while time.time() - start <= trainning_time:
        index = random.randint(0, len(files_list) - 1)
        logging.info("file: %s"  %  files_list[index][1])
        results = do_batch(files_list[index][0],network_start_func, network_end_func, reverse_start, reverse_end)
        stats = add_lists(stats, results[0])
        end_stats = add_lists(end_stats, results[1])
        i += 1
    logging.info("Did %d batches with %d bytes" % (i, i * BATCH_SIZE))
    return stats, end_stats

def test(files_list, network_start_func, network_end_func, reverse_start, reverse_end):
    stats = [0, 0, 0]
    end_stats = [0, 0, 0]
    for ep, fname in files_list:
        logging.info("file: %s"  %  fname)
        results = do_batch(ep, network_start_func, network_end_func, reverse_start, reverse_end, batch_size=ep.get_code_len())
        stats = add_lists(stats, results[0])
        end_stats = add_lists(end_stats, results[1])
    return stats, end_stats

def main(argv):

    parser = argparse.ArgumentParser(description='Binary functions recognization neural network')

    parser.add_argument('-d', "--data-dir", action="store", dest="data_dir", required=True)
    parser.add_argument('-x', '--test-percent', action="store", dest="test_percent", default=TEST_PERCENT, type=float)
    parser.add_argument('-m', "--load-model", action="store", dest="load_model")
    parser.add_argument('-s', "--save-model", action="store", dest="save_model")
    parser.add_argument('-l', "--log", action="store", dest="log_file")
    parser.add_argument('-t', "--time", action="store", dest="train_time", type=int, default=TRAINNING_TIME)
    parser.add_argument('-v', "--verbose", action="store_true", dest="verbose", default=False)
    parser.add_argument('-rs', "--reverse-start", action="store_true", dest="reverse_start", default=False)
    parser.add_argument('-ne', "--dont-reverse-end", action="store_false", dest="reverse_end", default=True)
    parser.add_argument('-r', "--learning-rate", action="store", type=float, dest="learning_rate", default=LEARNING_RATE)
    parser.add_argument('-c', "--converge-after", action="store", type=int, dest="converge_after", default=CONVERGE)


    args = parser.parse_args(argv[1:])
    
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if args.verbose:
        sh = logging.StreamHandler(sys.stdout)
        rootLogger.setLevel(logging.DEBUG)

        sh.setFormatter(format)
        rootLogger.addHandler(sh)
    if args.log_file != None:
        fh = logging.FileHandler(args.log_file, 'w')
        fh.setFormatter(format)
        rootLogger.addHandler(fh)
    is_train = args.test_percent < 1
    is_test = args.test_percent > 0
    logging.info("Dataset: %s | Reverse byte order - start: %s, end: %s" % (args.data_dir, args.reverse_start, args.reverse_end))
    if is_train:
        logging.info("Inital learning rate: %s | converge: %s | trainning time: %ss" % (args.learning_rate, args.converge_after, args.train_time))
    if is_test:
        logging.info("Testset percent: %s" % args.test_percent)
    logging.info("building network...")
    start_network = build_network()
    end_network = build_network()
    if args.load_model != None:
        load_model(start_network, end_network, args.load_model)
        logging.info("%s model loaded" % args.load_model)


    files = os.listdir(args.data_dir)
    train_set = []
    test_set = []

    for f in files:
        train_set.append([ElfParser(os.path.join(args.data_dir, f)), f])

    for i in xrange(int(math.ceil(args.test_percent * len(train_set)))):
        rand_index = random.randint(0, len(train_set) - 1)
        test_set.append(train_set.pop(rand_index))

    if is_train:
        network_start_func = build_func(start_network, True, args.learning_rate, args.converge_after)
        network_end_func = build_func(end_network, True, args.learning_rate, args.converge_after)
        stats, end_stats = train(train_set, network_start_func, network_end_func, args.reverse_start, args.reverse_end, args.train_time) 
        print_stats(stats, end_stats)

    if args.save_model != None:
        save_model(start_network, end_network, args.save_model)
        logging.info("model saved %s" % args.save_model)

    if is_test:
        logging.info("Start testing...")
        network_start_func = build_func(start_network, False)
        network_end_func = build_func(end_network, False)
        stats, end_stats = test(test_set, network_start_func, network_end_func, args.reverse_start, args.reverse_end) 
        print_stats(stats, end_stats)
    
if __name__ == "__main__":
    main(sys.argv)
