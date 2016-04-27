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
    rootLogger.setLevel(logging.DEBUG)
    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if args.verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format)
        rootLogger.addHandler(sh)
    if args.log_file != None:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(format)
        rootLogger.addHandler(fh)
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
        logging.info("%s: %d funcs, %d bytes" % (ep[1], len(funcs), len(text)))
        sum_f += len(funcs)
        sum_b += len(text)

    logging.info("Sum of funcs: %d, bytes: %d", (sum_f, sum_b))
    start = time.time()
    i = 0
    while time.time() - start <= args.train_time:
        index = random.randint(0, len(l) - 1)
        logging.info("file: " , l[index][1])
        do_batch(l[index][0],network_start_func, network_end_func, args.reverse)
        i += 1
    logging.info("Did %d batches with %d bytes" % (i, i * BATCH_SIZE))
    if args.save_model != None:
        save_model(start_network, end_network, args.save_model)
     

    


    


if __name__ == "__main__":
    main(sys.argv)
