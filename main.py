#!/usr/bin/python

from network import *
import os
import time
import sys
import argparse

class TrainParams(object):
    TIME_TRAINING = 0
    ITERATIONS_TRAINING = 1
    def __init__(self, train_type, value, end=None):
        self.type = train_type
        self.value = value
        self.end_value = end

def dummy_func(*args):
    return 0,0, [0] 

def dummy_finding_func(*args):
    return [0] 

def train(files_list, network_start_func, network_end_func, train_param):
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
    loss = []
    end_loss = []
    acc = []
    end_acc = []
    start_epoch = train_param.value
    end_epoch = train_param.end_value
    delta = time.time()
    curr = 0
    end = start_epoch if start_epoch > end_epoch else end_epoch
    if train_param.type == TrainParams.TIME_TRAINING:
        curr = delta
        end += delta
        start_epoch += delta
        if end_epoch is not None:
            end_epoch += delta

    while curr < end:
        s = ""
        files_for_batch = []
        for i in xrange(REAL_BATCH_SIZE):
            index = random.randint(0, len(files_list) - 1)
            s += "[%d] - %s," % (i, files_list[index][1])
            files_for_batch.append(files_list[index][0])
        logging.info("Epoch: %d | files: %s"  %  (curr, s))
        results = do_train_batch(files_for_batch, network_start_func, network_end_func)
        batch_stats = results[0]
        batch_end_stats = results[1]
        batch_loss = results[2]
        batch_end_loss = results[3]
        if network_start_func != dummy_func:
            stats = add_lists(stats, batch_stats)
        if network_end_func != dummy_func:
            end_stats = add_lists(end_stats, batch_end_stats)
        print_batch_stats(batch_stats, batch_end_stats, batch_loss, batch_end_loss)
        loss.append(batch_loss)
        end_loss.append(batch_end_loss)
        acc.append(calc_F1(stats))
        end_acc.append(calc_F1(end_stats))
        if train_param.type == TrainParams.ITERATIONS_TRAINING:
            curr += 1
        elif train_param.type == TrainParams.TIME_TRAINING:
            curr = time.time()
        if curr == start_epoch:
            logging.info("Finish start training")
            network_start_func = dummy_func
        elif end_epoch is not None and curr == end_epoch:
            logging.info("Finish end training")
            network_end_func = dummy_func


        
    return stats, end_stats, loss, end_loss, acc, end_acc

def find_functions(ep , network_start_func, network_end_func):
    logging.info("Starting find functions...")
    start, end = find_functions_in_file(ep, network_start_func, network_end_func) 
    for i in xrange(len(start)):
        if start[i] == 1:
            logging.info("func: %s" % ep.offset_to_va(i))
        if end[i] == 1:
            logging.info("end func: %s" % ep.offset_to_va(i))

def test(files_list, network_start_func, network_end_func):
    stats = [0, 0, 0]
    end_stats = [0, 0, 0]
    for ep, fname in files_list:
        logging.info("file: %s"  %  fname)
        results = do_test_batch(ep, network_start_func, network_end_func, batch_size=ep.get_code_len())
        stats = add_lists(stats, results[0])
        end_stats = add_lists(end_stats, results[1])
    return stats, end_stats

def main(argv):

    parser = argparse.ArgumentParser(description='Binary functions recognization neural network')

    parser.add_argument('-d', "--data-dir", action="store", dest="data_dir")
    parser.add_argument('-f', "--file", action="store", dest="file")
    parser.add_argument('-x', '--test-percent', action="store", dest="test_percent", default=TEST_PERCENT, type=float)
    parser.add_argument('-m', "--load-model", action="store", dest="load_model")
    parser.add_argument('-s', "--save-model", action="store", dest="save_model")
    parser.add_argument('-l', "--log", action="store_true", dest="log_file", default=False)
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument('-t', "--time", action="store", dest="train_time", type=int, default=TRAINING_TIME)
    train_group.add_argument('-i', "--iterations", action="store", dest="train_iterations", type=int, default=400)
    parser.add_argument('-e', "--end", action="store", dest="end_value", type=int)
    parser.add_argument('-v', "--verbose", action="store_true", dest="verbose", default=False)
    parser.add_argument('-r', "--learning-rate", action="store", type=float, dest="learning_rate", default=LEARNING_RATE)
    parser.add_argument('-c', "--converge-after", action="store", type=int, dest="converge_after", default=CONVERGE)


    args = parser.parse_args(argv[1:])
    if args.save_model != None:
        model_name = args.save_model
    elif args.load_model != None:
        model_name = args.load_model
    
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if args.verbose:
        sh = logging.StreamHandler(sys.stdout)
        rootLogger.setLevel(logging.DEBUG)

        sh.setFormatter(format)
        rootLogger.addHandler(sh)
    if args.log_file == True:
        fh = logging.FileHandler(model_name + ".log", 'w')
        fh.setFormatter(format)
        rootLogger.addHandler(fh)
    if args.file == None:
        is_train = args.test_percent < 1
        is_test = args.test_percent > 0
    else:
        is_train = False
        is_test = False
    logging.info("Batch size: %s. Before sequence len: %s. After sequence len: %s" % (BATCH_SIZE, SEQUENCE_BEFORE, SEQUENCE_AFTER)) 
    logging.info("Dataset: %s" % (args.data_dir if args.data_dir!=None else args.file))
    if is_train:
        if args.train_iterations != None:
            train_params = TrainParams(TrainParams.ITERATIONS_TRAINING, args.train_iterations, args.end_value)
            logging.info("training iterations: %s" % (args.train_iterations))
        elif args.train_time != None:
            train_params = TrainParams(TrainParams.TIME_TRAINING, args.train_time)
            logging.info("training time: %ss" % (args.train_time))
    if is_test:
        logging.info("Testset percent: %s" % args.test_percent)
    logging.info("building network...")
    start_network = build_network()
    end_network = build_network()
    #end_network = start_network 
    if args.load_model != None:
        load_model(start_network, end_network, args.load_model+".model")
        logging.info("%s.model model was loaded" % args.load_model)


    train_set = []
    test_set = []

    if is_train or is_test:
        files = os.listdir(args.data_dir)
        for f in files:
            train_set.append([ElfParser(os.path.join(args.data_dir, f)), f])

    if is_test:
        for i in xrange(int(math.ceil(args.test_percent * len(train_set)))):
            flag = True
            random.shuffle(train_set)
            for f in train_set:
                if f[0].get_code_len() <= 10000:
                    train_set.remove(f)
                    test_set.append(f)
                    flag = False
                    break
            if flag == True:
                break

    if is_train:
        try:
            network_start_func = build_train_func(start_network, args.learning_rate, args.converge_after)
            network_end_func = build_train_func(end_network, args.learning_rate, args.converge_after)
            # network_end_func = dummy_func 
            results = train(train_set, network_start_func, network_end_func, train_params) 
            stats, end_stats, loss, end_loss, acc, end_acc = results
        except:
            raise
        finally:
            logging.info("Did %d epochs with %d bytes" % (len(loss), len(loss) * BATCH_SIZE))
            draw_plot("epoch", "loss", loss, end_loss, model_name +"_loss.png", 1)
            draw_plot("epoch", "F1",  acc, end_acc, model_name+"_F1.png", 2)
            print_stats(stats, end_stats)
            if args.save_model != None:
                save_model(start_network, end_network, args.save_model+".model")
                logging.info("model was saved  to: %s.model" % args.save_model)

    if is_test:
        logging.info("Start testing...")
        network_start_func = build_test_func(start_network)
        network_end_func = build_test_func(end_network)
        # network_end_func = dummy_func
        stats, end_stats = test(test_set, network_start_func, network_end_func) 
        print_stats(stats, end_stats)

    if args.file:
        network_start_func = build_finding_func(start_network)
        network_end_func = build_finding_func(end_network)
        # network_end_func = dummy_finding_func 
        ep = ElfParser(args.file)
        find_functions(ep, network_start_func, network_end_func)  

if __name__ == "__main__":
    main(sys.argv)
