#!/usr/bin/python
"""
The Distr2Vec model
"""
from gensim import models
from copy import deepcopy
from math import ceil
import random
import numpy as np
import h5py
import os
import logging
import glob
import argparse
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("compute.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(fh)

iter_num = 10
lr = 0.05
train_uw2v = True


def remove_lastN_actions(plan, blank_count):
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    while cnt < blank_count:
        missing_action_index = len(plan)-1-cnt
        incomplete_plan[ missing_action_index ] = u'###'
        indices.append(missing_action_index)
        cnt += 1
    return blank_count, indices, incomplete_plan

def remove_conseq_middle_actions(plan, blank_count):
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    missing_action_init = random.randrange(2, len(plan) - 2 - blank_count)
    for i in range(missing_action_init, missing_action_init+blank_count):
        incomplete_plan[i] = u'###'
        indices.append(i)
        cnt += 1
    return blank_count, indices, incomplete_plan

def remove_randomN_actions(plan, blank_count):
    incomplete_plan = deepcopy(plan)
    indices = []
    cnt = 0
    while cnt < blank_count:
        missing_action_index = random.randrange(2, len(plan)-2)
        if missing_action_index in indices:
            # making sure that the indices generated are unique
            continue
        else:
            incomplete_plan[ missing_action_index ] = u'###'
            indices.append(missing_action_index)
            cnt += 1
    return blank_count, indices, incomplete_plan

def compute_fake_confidence_reliability(old_lambda, blankIndex, confidence):
    # confidence = list(confidence)
    reduced_lambda = reduce((lambda x, y: x * y), np.take(confidence, blankIndex))
    new_confidence = deepcopy(confidence)
    mean_confidence = np.mean(np.delete(new_confidence, blankIndex))
    confidence[blankIndex] = mean_confidence
    # new_lambda = reduce((lambda x, y: x * y), confidence)
    new_lambda = old_lambda*mean_confidence**len(blankIndex) / reduced_lambda
    return new_lambda, confidence

def EMTesting(weights, winSz, indices, incomplete_plan, blank_count, vocab_size, actions, topk, model, mode, lambda_i=0, confidence=0):
    update_flag = True
    for iter in range(iter_num):
        # save current_words to change update_flag
        current_words = []
        for index in indices:
            current_words.append(incomplete_plan[index])
        if update_flag:
            predict_words = []
            for blank_order in range(blank_count):
                tentative_plans = []
                blank_index = indices[blank_order]
                for vocab_index in range(vocab_size):
                    incomplete_plan[blank_index] = actions[vocab_index]
                    # build tmp_plan for compute score
                    tmp_plan = incomplete_plan[blank_index - winSz:blank_index + winSz + 1]
                    if mode == 'uw2v':
                        tmp_conf = confidence[blank_index - winSz:blank_index + winSz + 1]
                        tentative_plans.append((zip(tmp_plan,tmp_conf), lambda_i))
                    else: tentative_plans.append(tmp_plan)

                # if mode == 'uw2v':
                #     # # Zip incomplete_plan with confidence and reliability
                #     # new_tentative_plans = deepcopy(tentative_plans)
                #     # new_tentative_plans = zip(new_tentative_plans, [confidence[blank_index - winSz:blank_index + winSz + 1] * len(new_tentative_plans)])
                #     # new_tentative_plans = zip(new_tentative_plans, lambda_i * len(new_tentative_plans))
                #     scores_uw2v = model_uw2v.score(tentative_plans, mode) # should be a matrix?
                # else:
                scores = model.score(tentative_plans, mode)
                weights[:, blank_order] = scores
                # select word that has max score to update blank word
                max_index_uw2v = np.argmax(weights[:, blank_order])
                predict_word_uw2v = actions[max_index_uw2v]
                incomplete_plan[blank_index] = predict_word_uw2v
                predict_words.append(predict_word_uw2v)
            logger.info("predict words:%s", predict_words)
        else:
            # no update in the last iteration
            logger.info("quit from no update, iteration:%d", iter)
            break
        if predict_words == current_words:
            update_flag = False
        else:
            update_flag = True
    # if iter + 1 == iter_num:
    #     logger.info("quit from reach max iteration, iteration:%d", iter)
    best_plan_args = np.argsort(weights, axis=0)[-1 * topk:][::-1]
    return best_plan_args, weights

def compute_blank_count(missing, plan_length):
    if missing < 1:
        blank_count = int(ceil(plan_length * missing + 0.5))
    else: blank_count = int(missing)
    return blank_count

def obtain_incomplete_plan(mode, missing, plan, indices = None, blank_count = None):
    blank_count = compute_blank_count(missing, len(plan))
    if indices == None:
        if mode == 'end' and 0 < missing < 1:
            blank_count, indices, incomplete_plan = remove_lastN_actions(plan, blank_count)  # remove_lastN_actions(plan,1)
            indices.reverse()
        elif mode == 'end' and missing >= 1:
            blank_count, indices, incomplete_plan = remove_lastN_actions(plan, blank_count)
            indices.reverse()
        elif mode == 'middle_random' and missing >= 1:
            blank_count, indices, incomplete_plan = remove_randomN_actions(plan, blank_count)
        elif mode == 'middle_cons' and missing >= 1:
            blank_count, indices, incomplete_plan = remove_conseq_middle_actions(plan, blank_count)
        elif mode == 'middle_random' and 0 < missing < 1:
            blank_count, indices, incomplete_plan = remove_randomN_actions(plan, blank_count)
        elif mode == 'middle_cons' and 0 < missing < 1:
            blank_count, indices, incomplete_plan = remove_conseq_middle_actions(plan, blank_count)
        else:
            raise NotImplementedError
    else:
        incomplete_plan = deepcopy(plan)
        if type(indices) is int:
            incomplete_plan[indices] = u'###'
        elif type(indices) is list:
            for id in indices:
                incomplete_plan[id] = u'###'
    return blank_count, indices, incomplete_plan

class Distr2Vec(object):
    def __init__(self, gen_args, folderName, use_hs):
        self.folderName = folderName
        super(Distr2Vec, self).__init__()
        self.domain, self.shouldTrain, self.cvSplit, self.iter, self.topk, self.mode, self.missing, self.biasWin, self.num_train, self.winSz, self.beam_size, self.distr_sz = gen_args
        self.use_hs = use_hs
        dir = os.path.dirname(__file__)
        folder = os.path.join(dir) + '/' + self.domain + folderName
        self.files = glob.glob(folder + '*.h5')
        with h5py.File(self.files[0], 'r') as h5file:
            self.data_size = len(h5file['UncertainData'])
        if self.num_train is None:
            self.totalN = np.arange(self.data_size)
        else:
            self.totalN = np.arange(self.num_train)

    def train_and_test_core(self, testing_ids, testing_indices):
        if self.use_hs == 0: hs_list = [0]
        if self.use_hs == 1: hs_list = [1]
        if self.use_hs == 2: hs_list = [0, 1]
        train_times = []
        test_times = []
        acc = []

        for use_hs in hs_list:
            testing_ids = np.sort(testing_ids)
            training_ids = np.delete(self.totalN, testing_ids)
            # Train a model based on training data
            training_start_time = time.time()

            train_sentences_uw2v = models.uncertainWord2vec.DataLoader(self.domain, self.files, 'model2', self.num_train, self.beam_size,
                                                                       training_ids, distr_sz=self.distr_sz)
            model_uw2v = models.UncertainWord2Vec(uncertSentences=train_sentences_uw2v, min_count=1, sg=1,
                                                  workers=1, hs=use_hs, negative=0, window=self.winSz, iter=self.iter, sample=0, model=2)
            # model_uw2v.hs = 1
            model_uw2v.save(self.domain + '/model_M2' + '.txt')

            train_time = time.time() - training_start_time
            # print("--- training model %s needs %.2f seconds ---\n" % (self.__class__.__name__, train_time))

            plans = models.uncertainWord2vec.DataLoader(self.domain, self.files, 'model2', self.num_train, self.beam_size, testing_ids, distr_sz=self.distr_sz)  # Testing UDUP
            GP = models.uncertainWord2vec.DataLoader(self.domain, self.files, 'ground-truth', self.num_train, self.beam_size, testing_ids, distr_sz=self.distr_sz)
            GdPlans = [q for q in GP]
            NP = models.uncertainWord2vec.DataLoader(self.domain, self.files, 'certain-observation', self.num_train, self.beam_size, testing_ids, distr_sz=self.distr_sz)
            NoisyPlans = [p for p in NP]
            if train_uw2v == True:
                actions_uw2v = model_uw2v.wv.vocab.keys()
                vocab_size_uw2v = len(actions_uw2v)
                correct_uw2v = 0
            total = 0

            print "Distr2Vec Testing : Running on data %s with H-Softmax = %d" % (self.folderName, use_hs)
            testing_start_time = time.time()
            for itr, plan in enumerate(plans):
                # plan_grd = zip(*plan_grd)[0]
                plan_grd = zip(*GdPlans[itr])[0]
                # plan_dup = list(zip(*NoisyPlans[itr])[0])
                # blank_count, indices, incomplete_plan_dup = obtain_incomplete_plan(self.mode, self.missing, plan_dup)

                plan = list(plan)
                blank_count, indices, incomplete_plan_udup = obtain_incomplete_plan(self.mode, self.missing, plan, testing_indices[testing_ids[itr]]) # Yantian 051318, was testing_indices[itr]
                total += blank_count
                weights_uw2v = np.zeros(vocab_size_uw2v * blank_count).reshape(vocab_size_uw2v, blank_count)
                random_indices_uw2v = random.sample(range(vocab_size_uw2v), blank_count)  # random fill the blank word

                for order in range(blank_count):
                    blank_index = indices[order]
                    if train_uw2v == True:
                        random_word_uw2v = actions_uw2v[random_indices_uw2v[order]]
                        incomplete_plan_udup[blank_index] = random_word_uw2v

                best_plan_args_uw2v, weights_uw2v = EMTesting(weights_uw2v, self.winSz, indices, incomplete_plan_udup,
                                                              blank_count, vocab_size_uw2v, actions_uw2v, self.topk,
                                                              model_uw2v, 'uw2v2')

                if train_uw2v == True:
                    for blank_order in range(blank_count):
                        blank_index = indices[blank_order]
                        # for sample_index in best_plan_args_uw2v[:, blank_order]:
                        for sample_index in best_plan_args_uw2v[:, blank_order]:
                            if actions_uw2v[sample_index] == plan_grd[blank_index]:
                                correct_uw2v += 1
                                break
            test_time = time.time() - testing_start_time
            train_times.append(np.round(train_time, 2))
            test_times.append(np.round(test_time, 2))
            acc.append(np.round(float(correct_uw2v) / total, 4) * 100)
            total = 0
            correct_dup = 0
            # print("--- testing model %s needs %.2f seconds ---\n" % (self.__class__.__name__, test_time))

        return acc, train_times, test_times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_wv', type=bool, default=1, help='Need to train wordemb or not?')
    parser.add_argument('--cv_split', type=int, default=1, help='Number of cross validation blocks')
    parser.add_argument('--domain', type=str, default='blocks', help='domain?')
    parser.add_argument('--mode', type=str, default='end', help='middle or not')
    parser.add_argument('--num_missing', type=float, default=1, help='number of missing actions')
    parser.add_argument('--win_bias', type=int, default=0, help='window bias for word embedding')
    parser.add_argument('--num_train', type=int, default=0, help='number of training instances, 0 if use all')
    parser.add_argument('--beam_size', type=int, default=0, help='beam size, 0 if infinitely large')
    parser.add_argument('--iter', type=int, default=20, help='beam size, 0 if infinitely large')
    parser.add_argument('--top_k', nargs="+", type=int, help='range of size of candidate predictions: a b')
    parser.add_argument('--win_range', nargs="+", type=int, help='range of word2vec window size: a b')

    args = parser.parse_args()
    #print argv
    domain = args.domain
    cv_split = args.cv_split
    train = args.train_wv
    mode = args.mode
    num_missing = args.num_missing
    iter = args.iter
    biasWin = args.win_bias
    topkRange = tuple(args.top_k)
    winRange = tuple(args.win_range)
    num_train = (args.num_train == 0 and [None] or [args.num_train])[0]
    # beam_size = (args.beam_size == 0 and [None] or [args.beam_size])[0]
    beam_size = (args.beam_size == 0 and [np.inf] or [args.beam_size])[0]

    print "\n=== Domain : %s ===\n" % domain

    total_unknown_actions = 0
    total_correctUW2V_predictions = 0
    total_correctDUP_predictions = 0

    for topk in range(topkRange[0], topkRange[1] + 1):
        for winSz in range(winRange[0], winRange[1] + 1):
            gen_args = domain, train, cv_split, iter, topk, mode, num_missing, biasWin, num_train, winSz, beam_size
            # ua, cv, cp = train_and_test(gen_args)
            inst_e = EvaluatorM2(gen_args)
            ua, cv, cp = inst_e.run()
            total_unknown_actions += ua
            total_correctUW2V_predictions += cv
            total_correctDUP_predictions += cp

            print "\n==== FINAL STATISTICS ===="
            print "topk: %d" % (topk)
            print "window_size: %d" % (winSz)
            print "\nTotal unknown actions: %d; Total correct UW2V predictions: %0.2f; Total correct DUP predictions: %0.2f" % (total_unknown_actions, total_correctUW2V_predictions, total_correctDUP_predictions)
            print "UW2V ACCURACY: %0.2f%%\n" % (total_correctUW2V_predictions * 100.0 / total_unknown_actions)
            print "DUP ACCURACY: %0.2f%%\n" % (total_correctDUP_predictions * 100.0 / total_unknown_actions)

            total_unknown_actions = 0
            total_correctUW2V_predictions = 0
            total_correctDUP_predictions = 0

if __name__ == "__main__":
    # print compute_fake_confidence_reliability(0.0015, [1,3], np.array([0.1,0.1, 0.5, 0.3]).astype(float))
    main()

