#!/usr/bin/python
"""
The script for evaluating visual grounded plan corpora.
"""
from math import ceil, floor
import random
import sys
import numpy as np
import h5py
import os
import logging
import glob
import argparse
import ast

from U_train_and_test_RBM import RBM as Evaluator_RBM
from U_train_and_test_GM import GM as Evaluator_GM
from U_train_and_test_D2V import Distr2Vec as Evaluator_D2V

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("compute.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(fh)

def compute_blank_count(missing, plan_length):
    if missing < 1:
        blank_count = int(ceil(plan_length * missing + 0.5))
    else: blank_count = int(missing)
    return blank_count

class Evaluator(Evaluator_GM, Evaluator_RBM, Evaluator_D2V):
    def __init__(self, gen_args, folder, eval_hs):
        self.gen_args = gen_args
        self.folder = folder
        self.eval_hs = eval_hs
        # print self.folder
        self.testing_indices = []
        self.domain, self.shouldTrain, self.cvSplit, self.iter, self.topk, self.mode, self.missing, self.biasWin, self.num_train, self.winSz, self.beam_size, self.distr_sz = gen_args
        dir = os.path.dirname(__file__)
        folder = os.path.join(dir) + '/' + self.domain + self.folder
        self.files = glob.glob(folder + '*.h5')
        with h5py.File(self.files[0], 'r') as h5file:
            self.data_size = len(h5file['UncertainData'])
            for i in range(self.data_size):
                if self.domain == "video_salads":
                    seq_len = len(ast.literal_eval(h5file['UncertainData'][i]))
                else: seq_len = len(h5file['UncertainData'][i])
                blank_count = compute_blank_count(self.missing, seq_len)
                self.testing_indices.append(random.sample(range(seq_len), blank_count))  # The above cannot handle sequences with variable lengths
                assert np.all(np.greater(seq_len, self.testing_indices[-1])) == True, "seq_len is %d, last test indice is %d" % (seq_len, self.testing_indices[-1])

    def run(self):
        if self.num_train is None:
            self.totalN = np.arange(self.data_size)
        else:
            self.totalN = np.arange(self.num_train)
        shuffle_totalN = np.random.permutation(self.totalN)
        RandomTotal = np.split(shuffle_totalN, self.cvSplit)

        results = {}
        train_time = {}
        test_time = {}

        # use_hs = 3

        for i, testing_ids in enumerate(RandomTotal):
            for cls in Evaluator.__bases__:
                corrCV_pred = np.array([], dtype=np.float32)
                acc, trainTime, testTime = cls(self.gen_args, self.folder, self.eval_hs).train_and_test_core(testing_ids, self.testing_indices)

                if len(results)<len(Evaluator.__bases__):
                    corrCV_uw2v = np.append(corrCV_pred, acc)
                    results[cls.__name__] = [corrCV_uw2v]
                    train_time[cls.__name__] = [trainTime]
                    test_time[cls.__name__] = [testTime]
                else:
                    results[cls.__name__].append(np.asarray(acc))
                    train_time[cls.__name__].append(trainTime)
                    test_time[cls.__name__].append(testTime)
        return results, train_time, test_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_wv', type=bool, default=1, help='Need to train wordemb or not?')
    parser.add_argument('--cv_split', type=int, default=1, help='Number of cross validation blocks')
    parser.add_argument('--eval_hs', type=int, default=3, help='H-Softmax evaluation configure: 0 - only evaluate non-hs; 1 - only evaluate hs; 2 - evalute non-hs and then hs')
    parser.add_argument('--domain', type=str, default='blocks', help='domain?')
    parser.add_argument('--folders', nargs='+', type=str, default=['/distr_3_91100/'], help='data folder')
    parser.add_argument('--mode', type=str, default='end', help='middle or not')
    parser.add_argument('--num_missings', nargs='+', type=float, help='a list of values for number of missing observations. Note that inputing a number between 0 and 1 will be automatically understood as percentage (e.g. 0.1 0.2 0.3), otherwise please input integers, e.g. 1 2 3', required=True)
    parser.add_argument('--win_bias', type=int, default=0, help='window bias for word embedding')
    parser.add_argument('--num_train', type=int, default=0, help='number of training instances, 0 if use all')
    parser.add_argument('--beam_sizes', nargs='+', type=int, help='a list of beam size values, e.g. 1 3 5 7 9', required=True)
    parser.add_argument('--iter', type=int, default=20, help='beam size, 0 if infinitely large')
    parser.add_argument('--top_k', nargs="+", type=int, help='range of size of candidate predictions: a b')
    parser.add_argument('--win_range', nargs="+", type=int, help='range of word2vec window size: a b')

    args = parser.parse_args()
    domain = args.domain
    folders = args.folders
    cv_split = args.cv_split
    train = args.train_wv
    mode = args.mode
    num_missings = args.num_missings
    iter = args.iter
    biasWin = args.win_bias
    topkRange = tuple(args.top_k)
    winRange = tuple(args.win_range)
    num_train = (args.num_train == 0 and [None] or [args.num_train])[0]
    beam_sizes = (args.beam_sizes == 0 and [np.inf] or [args.beam_sizes])[0]
    eval_hs = args.eval_hs

    print "\n=== Domain : %s ===\n" % domain
    for topk in range(topkRange[0], topkRange[1] + 1):
        for beam_size in beam_sizes:
            for num_missing in num_missings: # 0.1, 0.2, 0.3, 0.4, 0.5
                for DD in folders: #[('/distr_3_91100/', 3)]:
                    for winSz in range(winRange[0], winRange[1] + 1):
                        folder, distr_sz = DD, 3
                        print "Evaluating dataset %s, with distribution size %d, %f missing actions, %d candidates, %d beam size, %d win_sz, %d recommendations per step, H-Softmax eval conf = %d." % (folder, distr_sz, num_missing, topk, beam_size, winSz, topk, eval_hs)

                        gen_args = domain, train, cv_split, iter, topk, mode, num_missing, biasWin, num_train, winSz, beam_size, distr_sz
                        inst_e = Evaluator(gen_args, folder, eval_hs)
                        results, train_time, test_time = inst_e.run()

                        for k, v in results.items():
                            for i, a in enumerate(zip(*v)):
                                print "Evaluating %s, if_using_hs is %d, Accuracy = %s\n" % (
                                k, i+(eval_hs%2) , np.round(np.mean(a), 2))
                        for k, t1 in train_time.items():
                            for i, t in enumerate(zip(*t1)):
                                print "Evaluating %s, with hs = %d. Training time: %f\n" % (
                                    k, i+(eval_hs%2), np.round(np.mean(t), 2))
                        for k, t2 in test_time.items():
                            for i, t in enumerate(zip(*t2)):
                                print "Evaluating %s, with hs = %d. Testing time: %f\n" % (
                                    k, i+(eval_hs%2), np.round(np.mean(t), 2))
if __name__=='__main__':
    main()

