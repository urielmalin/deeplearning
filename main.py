#!/usr/bin/python

from network import *
import os
import time
import sys
import argparse

def main(argv):

    parser = argparse.ArgumentParser(description='Binary functions recognization neural network')

    parser.add_argument('-d', "--data-dir", action="store", dest="data_dir", required=True)
    parser.add_argument('--train', action="store_true", dest="is_train", default=True, help="Train the network")
    parser.add_argument('--test', action="store_false", dest="is_train")
    parser.add_argument('-m', "--load-model", action="store", dest="load_model")
    parser.add_argument('-s', "--save-model", action="store", dest="save_model")
    parser.add_argument('-l', "--log", action="store", dest="log_file")
    parser.add_argument('-t', "--time", action="store", dest="train_time", type=int, default=TRAINNING_TIME)
    parser.add_argument('-v', "--verbose", action="store_true", dest="verbose", default=False)
    parser.add_argument('-n', "--dont-reverse", action="store_false", dest="reverse_start", default=True)


    args = parser.parse_args(argv[1:])
    
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if args.verbose:
        rootLogger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format)
        rootLogger.addHandler(sh)
    if args.log_file != None:
        fh = logging.FileHandler(args.log_file, 'w')
        fh.setFormatter(format)
        rootLogger.addHandler(fh)

    logging.info("building network...")
    start_network = build_network()
    end_network = build_network()
    if args.load_model != None:
        load_model(start_network, end_network, args.load_model)
        logging.info("model loaded")
    network_start_func = build_func(start_network, args.is_train)
    network_end_func = build_func(end_network, args.is_train)

    files = os.listdir(args.data_dir)
    l = []
    for f in files:
        l.append([ElfParser(os.path.join(args.data_dir, f)), f])

    stats = [0, 0, 0]
    end_stats = [0, 0, 0]
    if args.is_train:
        sum_f = 0 
        sum_b = 0
        for ep in l:
            text, funcs = ep[0].get_code_and_funcs()
            logging.info("%s: %d funcs, %d bytes" % (ep[1], len(funcs), len(text)))
            sum_f += len(funcs)
            sum_b += len(text)

        logging.info("Sum of funcs: %d, bytes: %d" % (sum_f, sum_b))

        start = time.time()
        i = 0
        while time.time() - start <= args.train_time:
            index = random.randint(0, len(l) - 1)
            logging.info("file: %s"  %  l[index][1])
            results = do_batch(l[index][0],network_start_func, network_end_func, args.reverse_start)
            stats = add_lists(stats, results[0])
            end_stats = add_lists(end_stats, results[1])
            i += 1
        logging.info("Did %d batches with %d bytes" % (i, i * BATCH_SIZE))
    else:
        for ep, fname in l:
            logging.info("file: %s"  %  fname)
            results = do_batch(ep, network_start_func, network_end_func, args.reverse_start, batch_size=ep.get_code_len())
            stats = add_lists(stats, results[0])
            end_stats = add_lists(end_stats, results[1])

    logging.info("start: %d TP, %d FP, %d FN" % (stats[0], stats[1], stats[2]))
    logging.info("Precision: %lf, Recall: %lf, F1: %lf" % (calc_precision(stats), calc_recall(stats), calc_F1(stats)))
    logging.info("end: %d TP, %d FP, %d FN" % (end_stats[0], end_stats[1], end_stats[2]))
    logging.info("Precision: %lf, Recall: %lf, F1: %lf" % (calc_precision(end_stats), calc_recall(end_stats), calc_F1(end_stats)))

    if args.save_model != None:
        save_model(start_network, end_network, args.save_model)
        logging.info("model saved")
     

    


    


if __name__ == "__main__":
    main(sys.argv)
