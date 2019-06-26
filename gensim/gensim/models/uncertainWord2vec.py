#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import ast
from copy import deepcopy
import numpy as np
import gc
import threading
import heapq
import itertools
from Queue import PriorityQueue
import warnings

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
from gensim.utils import keep_vocab_item, call_on_class_only
from gensim.models.keyedvectors import KeyedVectors, Vocab
import logging
from collections import defaultdict
from six import iteritems, itervalues, string_types
from six.moves import xrange
from timeit import default_timer
from types import GeneratorType
from scipy.special import expit
import cntk as C
import itertools
import sys

from Particle import ParticleFilter
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

logger = logging.getLogger(__name__)

try:
    from gensim.models.word2vec_inner import train_batch_sg as train_batch_sg_c  
    # from gensim.models.
    from gensim.models.word2vec_inner import score_sentence_sg_biasWin, score_sentence_cbow # score_sentence_sg,
    from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
    from uncertainWord2vec_inner import uncertain_train_batch_sg as uncertain_train_batch_sg_m1_c
    from gensim.models.word2vec_inner import train_batch_sg, train_batch_sg_biasWin, train_batch_cbow 

    raise ImportError
    # from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_sg_biasWin, score_sentence_cbow
    # from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    from gensim.models.word2vec_inner import score_sentence_sg
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000

    def train_batch_sg(model, sentences, alpha, work=None, compute_loss=False):
        """
        Update skip-gram model by training on a sequence of sentences.

        Each sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from `Word2Vec.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.

        """
        # print "naive"
        result = 0
        for sentence in sentences:
            word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                           model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        train_sg_pair(model, model.wv.index2word[word.index], word2.index, alpha, compute_loss=compute_loss)

            result += len(word_vocabs)
        return result

    def uncertain_train_batch_sg_m1(model, sentConfRels, alpha, work=None):
        """
        Train the uncertain word2vec with skip-gram algorithm.
        :param model:
        :param sentConfRels:
        :param alpha:
        :param work:
        :return:
        """
        result = 0
        # for (sentence, confidence, reliability) in zip(sentences, confidences, reliabilities):
        for compTrace in sentConfRels:
            # print sentence, confidence, reliability
            sentence = compTrace[0]
            confidence = compTrace[1]
            reliability = compTrace[2]
            assert all(isinstance(x, str) for x in sentence), (sentence, confidence, reliability)
            assert all(isinstance(x, np.float64) for x in confidence), (sentence, confidence, reliability)
                # print sentence
            # print sentence
            word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                           model.wv.vocab[w].sample_int > model.random.rand() * 2**32]
            # word_vocabs = [model.wv.vocab[w] for w in sentence]
            # for w in sentence:
            #     print "XXXXXXXX", w, model.wv.vocab[w]

            # Precompute latent vectors
            VL_TABLE = [[0 for x in range(len(word_vocabs))] for y in range(len(word_vocabs))]
            for pair in itertools.permutations(xrange(len(word_vocabs)),r=2):
                # print pair
                VL_TABLE[pair[0]][pair[1]] = np.random.normal(reliability, confidence[pair[0]]*confidence[pair[1]], len(model.wv.syn0))

            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    # print "pos2, word2", pos2, word2
                    if pos2 != pos:
                        # print "xxxx", confidence[pos]
                        # print confidence[pos], confidence[pos2]
                        confidMulp = confidence[pos]*confidence[pos2]
                        uncertain_train_sg_pair(VL_TABLE, pos, pos2, model, model.wv.index2word[word.index], word2.index, confidMulp, reliability, alpha)
            result += len(word_vocabs)
        return result

    def uncertain_train_batch_sg_m2(model, sentences, alpha, work=None):
        result = 0
        for uncerSentence in sentences:
            sentence_conf = zip(*uncerSentence)
            sentence_conf = [zip(*s) for s in sentence_conf]
            sentence, _ = zip(*sentence_conf)
            sentence = list(itertools.chain.from_iterable(sentence))
            # word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
            #                model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
            word_vocabs = deepcopy(uncerSentence)
            for pos, uncerWord in enumerate(uncerSentence):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        uncertain_train_sg_pair_m2(model, uncerWord, word2, alpha)
            result += len(word_vocabs)
        return result


    def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                      context_vectors=None, context_locks=None, compute_loss=False):
        if context_vectors is None:
            context_vectors = model.wv.syn0
        if context_locks is None:
            context_locks = model.syn0_lockf

        if word not in model.wv.vocab:
            return
        predict_word = model.wv.vocab[word]  # target word (NN output)

        l1 = context_vectors[context_index]  # input word (NN input/projection layer)
        lock_factor = context_locks[context_index]

        neu1e = zeros(l1.shape)

        if not (model.hs or model.negative):
            # l2c = deepcopy(model.syn_nv[predict_word.index])
            # prod_term = np.dot(l1, l2c.T)
            prod_term = np.dot(l1, model.syn_nv.T)
            # fc = model.softmax(prod_term)  # propagate hidden -> output
            fc = expit(prod_term)
            # print("BBB", prod_term)

            # print("len(model.wv.vocab)", len(model.wv.vocab))
            label = np.zeros([len(model.wv.vocab)], np.float64)
            label[predict_word.index] = 1
            gc = (label - fc) * alpha  # vector of error gradients multiplied by the learning rate
            # y = model.softmax(u)
            # neule = np.sum([np.subtract(y, word) for word in w_c], axis=0)
            if learn_hidden:
                # print("VVV", l1.shape, gc.shape, model.syn_nv.shape)
                model.syn_nv += outer(gc, l1)  # learn hidden -> output
            neu1e += np.dot(gc, model.syn_nv) # dot(gc, l2c)  # save error

            # if compute_loss:
            #
            #     lprob = -log(expit(-sgn * prod_term))
            #     model.running_training_loss += sum(lprob)

        if model.hs:
            # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
            l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
            prod_term = dot(l1, l2a.T)
            fa = expit(prod_term)  # propagate hidden -> output
            ga = (1 - predict_word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
            if learn_hidden:
                model.syn1[predict_word.point] += outer(ga, l1)  # learn hidden -> output
            neu1e += dot(ga, l2a)  # save error

            # loss component corresponding to hierarchical softmax
            if compute_loss:
                sgn = (-1.0) ** predict_word.code  # `ch` function, 0 -> 1, 1 -> -1
                lprob = -log(expit(-sgn * prod_term))
                model.running_training_loss += sum(lprob)

        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
                if w != predict_word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            prod_term = dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
            if learn_hidden:
                model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
            neu1e += dot(gb, l2b)  # save error

            # loss component corresponding to negative sampling
            if compute_loss:
                model.running_training_loss -= sum(log(expit(-1 * prod_term[1:])))  # for the sampled words
                model.running_training_loss -= log(expit(prod_term[0]))  # for the output word

        if learn_vectors:
            l1 += neu1e * lock_factor  # learn input -> hidden (mutates model.wv.syn0[word2.index], if that is l1)
        return neu1e

    def uncertain_train_sg_pair(VL_TABLE, pos, pos2, model, word, context_index, confidMulp, reliability, alpha, learn_vectors=True, learn_hidden=True,
                      context_vectors=None, context_locks=None):
        if context_vectors is None:
            W_E = model.wv.syn0 # W_E, i.e. word embedding matrix (pure knowledge)
        h_E = W_E[context_index]

        if context_locks is None:
            context_locks = model.syn0_lockf

        if word not in model.wv.vocab:
            return
        predict_word = model.wv.vocab[word]  # target word (NN output)
        neu1e = zeros(h_E.shape)

        # Init matrix W_{DEC}, bias b_{DEC}

        # Init matrix W_{LvH}, latent vector to hidden
        # W_LH = [np.random.randn(j, i) for i, j in zip(len(context_vectors), len(context_vectors[0]))]
        # W_LH = np.zeros(shape=(len(context_vectors), len(context_vectors[0])))

        # Sample latent vectors: x_i ~ N(L_v|reliability, confid_in*confid_targ)
        # reliability_sigma = 1.0/(1+expit(-1 * reliability))
        # confidMulp_sigma = 1.0/(1+expit(-1 * confidMulp))
        # VL = np.random.normal(1.0/reliability_sigma, 1.0/confidMulp_sigma, len(W_E))
        VL = np.random.normal(reliability, confidMulp, len(W_E))
        # VL = VL_TABLE[pos][pos2]

        # VL = np.random.normal(1.0/reliability, 1.0 / confidMulp, len(W_E))
        # print "VL", VL.shape
        # print "model.synDEC", model.synDEC.shape # (100, 95)

        # Forward compute v_ADJ, i.e. adjustment towards knowledge based on specific situations
        v_ADJ = np.matmul(model.wv.synDEC, VL)

        # Compute sigmoid(v_ADJ)
        # sigV_ADJ = 1 / (1 + np.exp(- v_ADJ)) # sigmoid activation
        tanhV_ADJ = np.tanh(v_ADJ) # tanh activation

        # Forward compute W_{CH}: l1 = W_{CH}[index] = W_E[index] + W_{LH}[index]
        # l1 = (W_E + W_LH)[context_index]
        # l1 = np.add(h_E, sigV_ADJ) # sigmoid activation
        l1 = np.add(h_E, tanhV_ADJ) # tanh activation
        # l1 = h_E * sigV_ADJ # Dot branch

        lock_factor = context_locks[context_index]

        if model.hs:
            # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
            l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
            fa = expit(dot(l1, l2a.T))  # propagate hidden -> output
            ga = (1 - predict_word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
            if learn_hidden:
                model.syn1[predict_word.point] += outer(ga, l1)  # learn hidden -> output
            neu1e += dot(ga, l2a)  # save error

        if model.negative:
            # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
                if w != predict_word.index:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            fb = expit(dot(l1, l2b.T))  # propagate hidden -> output
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
            if learn_hidden:
                model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
            neu1e += dot(gb, l2b)  # save error

        # print "neu1e", neu1e.shape
        # Update W_{DEC}
        # print "ubdex", word.index
        # delta_ADJ = neu1e*sigV_ADJ*(1-sigV_ADJ) # sigmoid activation
        delta_ADJ = neu1e*(1-tanhV_ADJ**2) # tanh activation
        # delta_ADJ = neu1e * h_E * sigV_ADJ * (1 - sigV_ADJ) # Dot branch loss

        # grad_DEC = -np.matmul(delta_LH, VL)*alpha
        grad_DEC = -outer(delta_ADJ, VL.T)*alpha
        model.wv.synDEC += grad_DEC

        if learn_vectors:
            # l1 = l1 + l_{W_{LH}}
            h_E += neu1e * lock_factor  # learn input -> hidden (mutates model.wv.syn0[word2.index], if that is l1)
            # h_E += neu1e * sigV_ADJ * lock_factor # Dot branch loss
        return neu1e

    def uncertain_train_sg_pair_m2(model, uncerWord, word2, alpha, learn_vectors=True, learn_hidden=True, context_vectors=None, context_locks=None):
        if context_vectors is None:
            context_vectors = model.wv.syn0
        if context_locks is None:
            context_locks = model.syn0_lockf

        l1 = zeros(model.vector_size)
        # label = np.zeros([len(model.wv.vocab)], np.float64)
        for (w,c) in uncerWord:
            l1 += float(c)*context_vectors[model.wv.vocab[w].index]  # input word (NN input/projection layer)
            # label[model.wv.vocab[w].index] = float(c)
        neu1e = zeros(l1.shape)

        if not (model.hs or model.negative):
            for (predict_word, confidence) in word2:
                predict_word = model.wv.vocab[predict_word]
                confidence = float(confidence)
                prod_term = np.dot(l1, model.syn_nv.T)
                # fc = model.softmax(prod_term)  # propagate hidden -> output
                fc = expit(prod_term)
                # print("BBB", prod_term)

                label = np.zeros([len(model.wv.vocab)], np.float64)
                label[predict_word.index] = 1
                gc = (label - fc) * alpha  # vector of error gradients multiplied by the learning rate
                # y = model.softmax(u)
                # neule = np.sum([np.subtract(y, word) for word in w_c], axis=0)
                if learn_hidden:
                    # print("VVV", l1.shape, gc.shape, model.syn_nv.shape)
                    model.syn_nv += confidence * outer(gc, l1)  # learn hidden -> output
                if learn_vectors:
                    neu1e += np.dot(gc, model.syn_nv) * confidence * context_locks[predict_word.index] # dot(gc, l2c)  # save error
        if model.hs:
            for (predict_word, confidence) in word2:
                confidence = float(confidence)
                predict_word = model.wv.vocab[predict_word]
                # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
                fa = expit(dot(l1, l2a.T))  # propagate hidden -> output
                ga = (1 - predict_word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                if learn_hidden:
                    model.syn1[predict_word.point] += confidence*outer(ga, l1)  # learn hidden -> output
                if learn_vectors:
                    neu1e += dot(ga, l2a) * confidence * context_locks[predict_word.index]  # save error

        if model.negative:
            for (predict_word, confidence) in word2:
                confidence = float(confidence)
                predict_word = model.wv.vocab[predict_word]
                # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
                word_indices = [predict_word.index]
                while len(word_indices) < model.negative + 1:
                    w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
                    if w != predict_word.index:
                        word_indices.append(w)
                l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
                fb = expit(dot(l1, l2b.T))  # propagate hidden -> output
                gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
                if learn_hidden:
                    model.syn1neg[word_indices] += outer(gb, l1)*confidence  # learn hidden -> output
                neu1e += dot(gb, l2b) * confidence * context_locks[predict_word.index]  # save error

        if learn_vectors:
            for (w, c) in uncerWord:
                # learn input -> hidden (mutates model.wv.syn0[word2.index], if that is l1)
                context_vectors[model.wv.vocab[w].index] += float(c)*neu1e
        return neu1e

    # def score_sentence_sg(model, sentence, work=None):
    #     """
    #     Obtain likelihood score for a single sentence in a fitted skip-gram representaion.
    #
    #     The sentence is a list of Vocab objects (or None, when the corresponding
    #     word is not in the vocabulary). Called internally from `Word2Vec.score()`.
    #
    #     This is the non-optimized, Python version. If you have cython installed, gensim
    #     will use the optimized version from word2vec_inner instead.
    #
    #     """
    #
    #     log_prob_sentence = 0.0
    #     # if model.negative: 
    #     if not model.hs:
    #         raise RuntimeError("scoring is only available for HS=True")
    #
    #     word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab]
    #     for pos, word in enumerate(word_vocabs):
    #         if word is None:
    #             continue  # OOV word in the input sentence => skip
    #
    #         # now go over all words from the window, predicting each one in turn
    #         start = max(0, pos - model.window)
    #         for pos2, word2 in enumerate(word_vocabs[start: pos + model.window + 1], start):
    #             # don't train on OOV words and on the `word` itself
    #             if word2 is not None and pos2 != pos:
    #                 log_prob_sentence += score_sg_pair(model, word, word2)
    #
    #     return log_prob_sentence

    def uncertain_score_sentence_sg(model, sentConfRels, work=None):
        """
        Obtain likelihood score for a single sentence in a fitted skip-gram representaion.

        The sentence is a list of Vocab objects (or None, when the corresponding
        word is not in the vocabulary). Called internally from `Word2Vec.score()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.

        """
        conf_sent, lambda_i = sentConfRels
        sentence, confidence = zip(*conf_sent)

        log_prob_sentence = 0.0
        # if model.negative:
        if not model.hs:
            raise RuntimeError("scoring is only available for HS=True")

        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab]
        # Precompute latent vectors
        VL_TABLE = [[0 for x in range(len(word_vocabs))] for y in range(len(word_vocabs))]
        for pair in itertools.permutations(xrange(len(word_vocabs)), r=2):
            # print pair
            VL_TABLE[pair[0]][pair[1]] = np.random.normal(lambda_i, confidence[pair[0]] * confidence[pair[1]],
                                                          len(model.wv.syn0))

        for pos, word in enumerate(word_vocabs):
            if word is None:
                continue  # OOV word in the input sentence => skip

            # now go over all words from the window, predicting each one in turn
            start = max(0, pos - model.window)
            for pos2, word2 in enumerate(word_vocabs[start: pos + model.window + 1], start):
                # don't train on OOV words and on the `word` itself
                if word2 is not None and pos2 != pos:
                    confidMulp = confidence[pos]*confidence[pos2]
                    log_prob_sentence += uncertain_score_sg_pair(VL_TABLE, pos, pos2, model, word, word2, confidMulp, lambda_i)

        return log_prob_sentence

    def uncertain_score_sentence_sg_m2(model, uncerSentence, work=None):
        log_prob_sentence = 0.0
        # if model.negative: 
        # if not model.hs:
        #     raise RuntimeError("scoring is only available for HS=True")

        word_vocabs = deepcopy(uncerSentence)

        for pos, word in enumerate(uncerSentence):
            if word is None:
                continue  # OOV word in the input sentence => skip

            # now go over all words from the window, predicting each one in turn
            start = max(0, pos - model.window)
            for pos2, word2 in enumerate(word_vocabs[start: pos + model.window + 1], start):
                # don't train on OOV words and on the `word` itself
                if word2 is not None and pos2 != pos:
                    log_prob_sentence += uncertain_score_sg_pair_m2(model, word, word2)

        return log_prob_sentence

    def score_sg_pair(model, word, word2):
        if model.hs:
            l1 = model.wv.syn0[word2.index]
            l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
            sgn = (-1.0) ** word.code  # ch function, 0-> 1, 1 -> -1
            lprob = -logaddexp(0, -sgn * dot(l1, l2a.T))
            return sum(lprob)
        else:
            l1 = model.wv.syn0[word2.index]
            l2a = deepcopy(model.syn1[word.index])
            lprob = -logaddexp(0, dot(l1, l2a.T))
            return lprob

def uncertain_score_sg_pair(VL_TABLE, pos, pos2, model, word, word2, confidMulp, reliability):
        W_E = model.wv.syn0
        h_E = W_E[word2.index]
        # Sampling
        # VL = np.random.normal(1.0/reliability, 1.0/confidMulp, len(W_E))
        # reliability_sigma = 1.0 / (1 + expit(-1 * reliability))
        # confidMulp_sigma = 1.0 / (1 + expit(-1 * confidMulp))
        # VL = np.random.normal(1.0/reliability_sigma, 1.0 / confidMulp_sigma, len(W_E))
        VL = np.random.normal(reliability, confidMulp, len(W_E))
        # VL = VL_TABLE[pos][pos2]

        # Forward compute v_ADJ, i.e. adjustment towards knowledge based on specific situations
        v_ADJ = np.matmul(model.wv.synDEC, VL)
        # Compute sigmoid(v_ADJ)
        # sigV_ADJ = 1 / (1 + np.exp(- v_ADJ)) # sigmoid activation
        tanhV_ADJ = np.tanh(v_ADJ)  # tanh activation
        # Forward compute W_{CH}: l1 = W_{CH}[index] = W_E[index] + W_{LH}[index]
        # l1 = (W_E + W_LH)[context_index]
        # l1 = np.add(h_E, sigV_ADJ) # sigmoid activation
        l1 = np.add(h_E, tanhV_ADJ) # tanh activation
                    # l1 = h_E * sigV_ADJ

        l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
        sgn = (-1.0) ** word.code  # ch function, 0-> 1, 1 -> -1
        lprob = -logaddexp(0, -sgn * dot(l1, l2a.T))
        return sum(lprob)

def uncertain_score_sg_pair_m2(model, word, word2):
    l1 = zeros(model.vector_size)
    if isinstance(word2, basestring):
        if word2 not in model.wv.vocab:
            return 0
        word2 = model.wv.vocab[word2]
        l1 = model.wv.syn0[word2.index]
    elif isinstance(word2, tuple):
        for (w, c) in word2:
            if w not in model.wv.vocab:
                continue
            l1 += float(c) * model.wv.syn0[model.wv.vocab[w].index]  # input word (NN input/projection layer)
    if isinstance(word, basestring):
        if word not in model.wv.vocab:
            return 0
        predict_word = model.wv.vocab[word]
        if model.hs:
            l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
            sgn = (-1.0) ** predict_word.code  # ch function, 0-> 1, 1 -> -1
            lprob = -sum(logaddexp(0, -sgn * dot(l1, l2a.T)))
        else:
            l2c = deepcopy(model.syn_nv[predict_word.index])
            lprob = -logaddexp(0, -dot(l1, l2c.T))
    elif isinstance(word, tuple):
        lprob = 0
        for (predict_word, confidence) in word:
            if predict_word not in model.wv.vocab:
                continue
            confidence = float(confidence)
            predict_word = model.wv.vocab[predict_word]
            if model.hs:
                l2a = deepcopy(model.syn1[predict_word.point])
                sgn = (-1.0) ** predict_word.code
                # print "CCC",l2a.shape
                lprob += -sum(logaddexp(0, -sgn * dot(l1, l2a.T))) * confidence
            else:
                l2a = deepcopy(model.syn_nv[predict_word.index])
                # print "CCC",l2a.shape
                lprob += -logaddexp(0, -dot(l1, l2a.T)) * confidence
    return lprob

class UncertainWord2Vec(utils.SaveLoad):

    def __init__(
            self, uncertSentences=None, uncertainTrain = True, size=100, alpha=0.025, window=5, biasWin=0, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, model=1):
        self.uncertainTrain = uncertainTrain
        self.initialize_word_vectors()
        self.vector_size = int(size)
        self.layer1_size = int(size)
        self.window = int(window)
        self.biasWin = int(biasWin)
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.sample = sample
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.model_trimmed_post_training = False
        self.workers = int(workers)
        self.alpha = float(alpha)
        self.iter = iter 
        self.min_alpha = float(min_alpha)
        self.batch_words = batch_words
        self.train_count = 0
        self.sorted_vocab = sorted_vocab
        self.seed = seed
        self.hashfxn = hashfxn
        self.null_word = null_word
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.random = random.RandomState(seed)
        self.total_train_time = 0
        self.model = model

        if self.model == 1:
            uncertSentences = list(uncertSentences)
            reliabilities, confidences_sentences = zip(*uncertSentences)
            confidences_sentences = [zip(*x) for x in confidences_sentences]
            sentences, confidences = zip(*confidences_sentences)
            data = sentences, confidences, reliabilities
            if sentences is not None:
                self.build_vocab(sentences, trim_rule=trim_rule)
                # Old train
                if self.uncertainTrain == True:
                    self.train(data)
                else:
                    self.pure_train(sentences)
        elif self.model == 2:
            uncertSentences = list(uncertSentences)
            confidences_sentences = zip(*uncertSentences[0])
            confidences_sentences = [zip(*x) for x in confidences_sentences]
            sentences, confidences = zip(*confidences_sentences)
            if sentences is not None:
                self.build_vocab(uncertSentences, trim_rule=trim_rule)
                self.train_model2(uncertSentences)

        elif self.model == 3:
            # if isinstance(uncertSentences, GeneratorType):
            #     raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")

            uncertSentences = list(uncertSentences)
            # confidences_sentences = zip(*uncertSentences)
            confidences_sentences = [zip(*x) for x in uncertSentences]
            sentences, confidences = zip(*confidences_sentences)
            if sentences is not None:
                self.build_vocab(sentences, trim_rule=trim_rule)
                # Old train
                self.pure_train(sentences)

            # uncertSentences = list(uncertSentences)
            # confidences_sentences = zip(*uncertSentences)
            # confidences_sentences = [zip(*x) for x in confidences_sentences]
            # sentences, confidences = zip(*confidences_sentences)
            # if sentences is not None:
            #     self.build_vocab(sentences, trim_rule=trim_rule)
            #     self.pure_train(sentences)

    def initialize_word_vectors(self):
        self.wv = KeyedVectors()

    # def train(self, sentences, confidences, reliabilities, total_words=None, word_count=0,
    #           total_examples=None, queue_factor=2, report_delay=1.0): 
    def train(self, data, total_words=None, word_count=0,
                  total_examples=None, queue_factor=2, report_delay=1.0):  
        if (self.model_trimmed_post_training):
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s window=%s",
            self.workers, len(self.wv.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative, self.window)

        if not self.wv.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.syn0):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if not hasattr(self, 'corpus_count'):
            raise ValueError(
                "The number of sentences in the training corpus is missing. Did you load the model via KeyedVectors.load_word2vec_format?"
                "Models loaded via load_word2vec_format don't support further training. "
                "Instead start with a blank model, scan_vocab on the new corpus, intersect_word2vec_format with the old model, then train.")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i sentences, matching count from corpus used for vocabulary survey",
                            total_examples)
            else:
                raise ValueError(
                    "you must provide either total_words or total_examples, to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            # sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            # confidences = utils.RepeatCorpusNTimes(confidences, self.iter)
            # reliabilities = utils.RepeatCorpusNTimes(reliabilities, self.iter)
            data = utils.RepeatUncertainCorpusNTimes(data, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL) 
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                # sentences, confidences, reliabilities, alpha = job
                sentConfRel, alpha = job
                # tally, raw_tally = self._do_train_job(sentConfRel[0], sentConfRel[1], sentConfRel[2], alpha, (work, neu1))  # interface to training
                tally, raw_tally = self._do_train_job_uncertain(sentConfRel, alpha, (work, neu1))
                progress_queue.put((len(sentConfRel[0]), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warn("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = next_alpha
            job_no = 0

            # c_ = np.array(data.corpus[1])
            # confidences = c_.astype(np.float)
            # r_ = np.array(data.corpus[2])
            # reliabilities = r_.astype(np.float)

            # for id, (sent, conf, reli) in enumerate(zip(data.corpus[0], confidences, reliabilities)):
            for id, (sent, conf, reli) in enumerate(data):
                sentence_length = self._raw_word_count([sent])
                assert all(isinstance(x, str) for x in sent), sent
                assert all(isinstance(x, np.float64) for x in conf), conf

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append((sent, conf, reli))
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_alpha)
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if self.min_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                        next_alpha = max(self.min_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [(sent, conf, reli)], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warn("under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    def pure_train(self, sentences, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        if (self.model_trimmed_post_training):
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s window=%s",
            self.workers, len(self.wv.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative, self.window)

        if not self.wv.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.syn0):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if not hasattr(self, 'corpus_count'):
            raise ValueError(
                "The number of sentences in the training corpus is missing. Did you load the model via KeyedVectors.load_word2vec_format?"
                "Models loaded via load_word2vec_format don't support further training. "
                "Instead start with a blank model, scan_vocab on the new corpus, intersect_word2vec_format with the old model, then train.")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i sentences, matching count from corpus used for vocabulary survey", total_examples)
            else:
                raise ValueError("you must provide either total_words or total_examples, to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, alpha = job
                tally, raw_tally = self._do_train_job(sentences, alpha, (work, neu1)) #  interface to training
                progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warn("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = next_alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_alpha)
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if self.min_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                        next_alpha = max(self.min_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warn("under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    def train_model2(self, sentences, total_words=None, word_count=0,
              total_examples=None, queue_factor=2, report_delay=1.0):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        if (self.model_trimmed_post_training):
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s window=%s",
            self.workers, len(self.wv.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative, self.window)

        if not self.wv.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.syn0):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if not hasattr(self, 'corpus_count'):
            raise ValueError(
                "The number of sentences in the training corpus is missing. Did you load the model via KeyedVectors.load_word2vec_format?"
                "Models loaded via load_word2vec_format don't support further training. "
                "Instead start with a blank model, scan_vocab on the new corpus, intersect_word2vec_format with the old model, then train.")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i sentences, matching count from corpus used for vocabulary survey", total_examples)
            else:
                raise ValueError("you must provide either total_words or total_examples, to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, alpha = job
                tally, raw_tally = self._do_train_job_uncertain(sentences, alpha, (work, neu1)) # interface to training
                progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            if next_alpha > self.min_alpha_yet_reached:
                logger.warn("Effective 'alpha' higher than previous training cycles")
            self.min_alpha_yet_reached = next_alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                        job_no, batch_size, len(job_batch), next_alpha)
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if self.min_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                        next_alpha = max(self.min_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warn("under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    def score(self, sentences, mode, total_sentences=int(1e6), chunksize=100, queue_factor=2, report_delay=1):
        """
        Score the log probability for a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
        This does not change the fitted model in any way (see Word2Vec.train() for that).

        We have currently only implemented score for the hierarchical softmax scheme,
        so you need to have run word2vec with hs=1 and negative=0 for this to work.

        Note that you should specify total_sentences; we'll run into problems if you ask to
        score more than this number of sentences but it is inefficient to set the value too high.

        See the article by [taddy]_ and the gensim demo at [deepir]_ for examples of how to use such scores in document classification.

        .. [taddy] Taddy, Matt.  Document Classification by Inversion of Distributed Language Representations, in Proceedings of the 2015 Conference of the Association of Computational Linguistics.
        .. [deepir] https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb

        """
        # print("GGGGGG", sentences)
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension compilation failed, scoring will be slow. "
                          "Install a C compiler and reinstall gensim for fastness.")

        logger.info(
            "scoring sentences with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s and negative=%s",
            self.workers, len(self.wv.vocab), self.layer1_size, self.sg, self.hs, self.sample, self.negative)

        if not self.wv.vocab:
            raise RuntimeError("you must first build vocabulary before scoring new data")

        # if not self.hs:
        #     raise RuntimeError("We have currently only implemented score \
        #             for the hierarchical softmax scheme, so you need to have \
        #             run word2vec with hs=1 and negative=0 for this to work.")

        def worker_loop():
            """Compute log probability for each sentence, lifting lists of sentences from the jobs queue."""
            work = zeros(1, dtype=REAL)  # for sg hs, we actually only need one memory loc (running sum)
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            while True:
                job = job_queue.get()
                if job is None:  # signal to finish
                    break
                ns = 0
                for sentence_id, sentence in job:
                    if sentence_id >= total_sentences:
                        break
                    if self.sg:
                        #if self.biasWin == 0:
                        if mode == 'uw2v':
                            score = uncertain_score_sentence_sg(self, sentence, work) #  interface
                        elif mode == 'dup':
                            score = score_sentence_sg(self, sentence, work)
                        elif mode == 'uw2v2':
                            score = uncertain_score_sentence_sg_m2(self, sentence, work)
                       # else: score = score_sentence_sg_biasWin(self, sentence, work)
                    else:
                        score = score_sentence_cbow(self, sentence, work, neu1)
                    sentence_scores[sentence_id] = score
                    ns += 1
                progress_queue.put(ns)  # report progress

        start, next_report = default_timer(), 1.0
        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        sentence_count = 0
        sentence_scores = matutils.zeros_aligned(total_sentences, dtype=REAL)

        push_done = False
        done_jobs = 0
        jobs_source = enumerate(utils.grouper(enumerate(sentences), chunksize))

        # fill jobs queue with (id, sentence) job items
        while True:
            try:
                job_no, items = next(jobs_source)
                if (job_no - 1) * chunksize > total_sentences:
                    logger.warning(
                        "terminating after %i sentences (set higher total_sentences if you want more).",
                        total_sentences)
                    job_no -= 1
                    raise StopIteration()
                logger.debug("putting job #%i in the queue", job_no)
                job_queue.put(items)
            except StopIteration:
                logger.info(
                    "reached end of input; waiting to finish %i outstanding jobs",
                    job_no - done_jobs + 1)
                for _ in xrange(self.workers):
                    job_queue.put(None)  # give the workers heads up that they can finish -- no more work!
                push_done = True
            try:
                while done_jobs < (job_no + 1) or not push_done:
                    ns = progress_queue.get(push_done)  # only block after all jobs pushed
                    sentence_count += ns
                    done_jobs += 1
                    elapsed = default_timer() - start
                    if elapsed >= next_report:
                        logger.info(
                            "PROGRESS: at %.2f%% sentences, %.0f sentences/s",
                            100.0 * sentence_count, sentence_count / elapsed)
                        next_report = elapsed + report_delay  # don't flood log, wait report_delay seconds
                else:
                    # loop ended by job count; really done
                    break
            except Empty:
                pass  # already out of loop; continue to next push

        elapsed = default_timer() - start
        self.clear_sims()
        logger.info(
            "scoring %i sentences took %.1fs, %.0f sentences/s",
            sentence_count, elapsed, sentence_count / elapsed)
        return sentence_scores[:sentence_count]

    def clear_sims(self):
        self.wv.syn0norm = None

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, update=update)  # trim by min_count & precalculate downsampling
        self.finalize_vocab(update=update)  # build tables & arrays

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None, model=1):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            if not checked_string_types:
                if isinstance(sentence, string_types):
                    logger.warn("Each 'sentences' item should be a list of words (usually unicode strings)."
                                "First item here is instead plain %s.", type(sentence))
                checked_string_types += 1
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                            sentence_no, sum(itervalues(vocab)) + total_words, len(vocab))

            if self.model in (1, 3):
                for word in sentence:
                    vocab[word] += 1
            elif self.model == 2:
                for word in sentence:
                    for w in word:
                        vocab[w[0]] += 1

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                total_words += utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        total_words += sum(itervalues(vocab))
        logger.info("collected %i word types from a corpus of %i raw words and %i sentences",
                    len(vocab), total_words, sentence_no + 1)
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab

    def scale_vocab(self, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False, trim_rule=None, update=False):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        if not update:
            logger.info("Loading a fresh vocabulary")
            retain_total, retain_words = 0, []
            # Discard words less-frequent than min_count
            if not dry_run:
                self.wv.index2word = []
                # make stored settings match these applied settings
                self.min_count = min_count
                self.sample = sample
                self.wv.vocab = {}

            for word, v in iteritems(self.raw_vocab):
                if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                    retain_words.append(word)
                    retain_total += v
                    if not dry_run:
                        self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                        self.wv.index2word.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            original_unique_total = len(retain_words) + drop_unique
            retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1)
            logger.info("min_count=%d retains %i unique words (%i%% of original %i, drops %i)",
                        min_count, len(retain_words), retain_unique_pct, original_unique_total, drop_unique)
            original_total = retain_total + drop_total
            retain_pct = retain_total * 100 / max(original_total, 1)
            logger.info("min_count=%d leaves %i word corpus (%i%% of original %i, drops %i)",
                        min_count, retain_total, retain_pct, original_total, drop_total)
        else:
            logger.info("Updating model with new vocabulary")
            new_total = pre_exist_total = 0
            new_words = pre_exist_words = []
            for word, v in iteritems(self.raw_vocab):
                if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                    if word in self.wv.vocab:
                        pre_exist_words.append(word)
                        pre_exist_total += v
                        if not dry_run:
                            self.wv.vocab[word].count += v
                    else:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                            self.wv.index2word.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            original_unique_total = len(pre_exist_words) + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            logger.info("""New added %i unique words (%i%% of original %i)
                        and increased the count of %i pre-existing words (%i%% of original %i)""",
                        len(new_words), new_unique_pct, original_unique_total,
                        len(pre_exist_words), pre_exist_unique_pct, original_unique_total)
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.wv.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_unique, 'retain_total': retain_total,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}

        # print extra memory estimates
        report_values['memory'] = self.estimate_memory(vocab_size=len(retain_words))

        return report_values

    def finalize_vocab(self, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.wv.index2word:
            self.scale_vocab()
        if self.sorted_vocab and not update:
            self.sort_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.wv.vocab)
            self.wv.index2word.append(word)
            self.wv.vocab[word] = v
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights()
        else:
            self.update_weights()

    def sort_vocab(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if len(self.wv.syn0):
            raise RuntimeError("cannot sort vocabulary after model weights already initialized.")
        self.wv.index2word.sort(key=lambda word: self.wv.vocab[word].count, reverse=True)
        for i, word in enumerate(self.wv.index2word):
            self.wv.vocab[word].index = i

    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words", len(self.wv.vocab))

        # build the huffman tree
        heap = list(itervalues(self.wv.vocab))
        heapq.heapify(heap)
        for i in xrange(len(self.wv.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.wv.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.wv.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.wv.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i", max_depth)

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        if not (self.hs or self.negative):
            self.syn_nv = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
        if self.uncertainTrain == True:
            fan_in = len(self.wv.vocab)
            fan_out = self.vector_size
            stddevDEC = sqrt(2.0 / (fan_in + fan_out))

            self.wv.syn0 = empty((len(self.wv.vocab), self.vector_size), dtype=REAL) # The matrix of word embeddings
            # self.wv.synDEC = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)
            self.wv.synDEC = C.parameter((fan_out, fan_in), init=C.initializer.truncated_normal(stddevDEC), dtype=np.float32).value
            # self.wv.syn2 =

            # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
            for i in xrange(len(self.wv.vocab)):
                # construct deterministic seed from word AND seed argument
                self.wv.syn0[i] = self.seeded_vector(self.wv.index2word[i] + str(self.seed))
            # for i in xrange(len(self.wv.vocab)):
            #     # construct deterministic seed from word AND seed argument
            #     self.wv.synDEC[i] = np.random.rand(self.vector_size)
            # self.wv.synDEC = np.transpose(self.wv.synDEC)
            if self.hs:
                self.syn1 = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
            if self.negative:
                self.syn1neg = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)

            self.wv.syn0norm = None

            # self.synDEC = zeros((self.layer1_size, len(self.wv.vocab)), dtype=REAL) # TODO, use random number?
            self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning
        else:
            # self.wv.syn0 = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)  # The matrix of word embeddings
            # # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
            # for i in xrange(len(self.wv.vocab)):
            #     # construct deterministic seed from word AND seed argument
            #     self.wv.syn0[i] = self.seeded_vector(self.wv.index2word[i] + str(self.seed))

            self.wv.syn0 = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)
            for i in xrange(len(self.wv.vocab)):
                # construct deterministic seed from word AND seed argument
                self.wv.syn0[i] = self.seeded_vector(self.wv.index2word[i] + str(self.seed))
            if self.hs:
                self.syn1 = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
            if self.negative:
                self.syn1neg = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
            self.wv.syn0norm = None
            # self.synDEC = zeros((self.layer1_size, len(self.wv.vocab)), dtype=REAL) # TODO, use random number?
            self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning

    def update_weights(self):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        logger.info("updating layer weights")
        gained_vocab = len(self.wv.vocab) - len(self.wv.syn0)
        newsyn0 = empty((gained_vocab, self.vector_size), dtype=REAL)

        # randomize the remaining words
        for i in xrange(len(self.wv.syn0), len(self.wv.vocab)):
            # construct deterministic seed from word AND seed argument
            newsyn0[i-len(self.wv.syn0)] = self.seeded_vector(self.wv.index2word[i] + str(self.seed))

        # Raise an error if an online update is run before initial training on a corpus
        if not len(self.wv.syn0):
            raise RuntimeError("You cannot do an online vocabulary-update of a model which has no prior vocabulary. " \
                "First build the vocabulary of your model with a corpus " \
                "before doing an online update.")

        self.wv.syn0 = vstack([self.wv.syn0, newsyn0])
        self.wv.synDEC = vstack([self.synDEC, zeros((gained_vocab, self.layer1_size), dtype=REAL)])

        if self.hs:
            self.syn1 = vstack([self.syn1, zeros((gained_vocab, self.layer1_size), dtype=REAL)])
        if self.negative:
            self.syn1neg = vstack([self.syn1neg, zeros((gained_vocab, self.layer1_size), dtype=REAL)])

        if not (self.hs or self.negative):
            self.syn_nv = vstack([self.syn_nv, zeros((gained_vocab, self.layer1_size), dtype=REAL)])

        self.wv.syn0norm = None
        # self.synDEC =  # TODO
        # do not suppress learning for already learned words
        self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(self.vector_size) - 0.5) / self.vector_size

    def reset_from(self, other_model):
        """
        Borrow shareable pre-built structures (like vocab) from the other_model. Useful
        if testing multiple models in parallel on the same corpus.
        """
        self.wv.vocab = other_model.vocab
        self.wv.index2word = other_model.index2word
        self.cum_table = other_model.cum_table
        self.corpus_count = other_model.corpus_count
        self.reset_weights()

    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.wv.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in xrange(vocab_size):
            train_words_pow += self.wv.vocab[self.wv.index2word[word_index]].count**power
        cumulative = 0.0
        for word_index in xrange(vocab_size):
            cumulative += self.wv.vocab[self.wv.index2word[word_index]].count**power
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def _do_train_job(self, sentConfRels, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentConfRels, alpha, work)
            # tally += train_batch_sg_c(self, sentConfRels, alpha, work)
            # tally += train_batch_sg(self, sentConfRels, alpha, work)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)
        sentences = [sentConfRels[i][0] for i in xrange(len(sentConfRels))] # TODO fix
        return tally, self._raw_word_count(sentences)

    def _do_train_job_uncertain(self, sentConfRels, alpha, inits):
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            if self.model == 1:
                # tally += uncertain_train_batch_sg_m1(self, sentConfRels, alpha, work)
                tally += uncertain_train_batch_sg_m1_c(self, sentConfRels, alpha, work)
            elif self.model == 2:
                tally += uncertain_train_batch_sg_m2(self, sentConfRels, alpha, work)
            else: raise NotImplementedError
        else:
            tally += train_batch_cbow(self, sentConfRels, alpha, work, neu1)
        if self.model == 1:
            sentConfRels = [sentConfRels[i][0] for i in xrange(len(sentConfRels))]
        return tally, self._raw_word_count(sentConfRels)

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return sum(len(sentence) for sentence in job)

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings and provided vocabulary size."""
        vocab_size = vocab_size or len(self.wv.vocab)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['syn0'] = vocab_size * self.vector_size * dtype(REAL).itemsize
        if self.hs:
            report['syn1'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.layer1_size * dtype(REAL).itemsize
        report['total'] = sum(report.values())
        logger.info("estimated required memory for %i words and %i dimensions: %i bytes",
                    vocab_size, self.vector_size, report['total'])

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors, recalculable table
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm', 'table', 'cum_table'])

        super(UncertainWord2Vec, self).save(*args, **kwargs)

    save.__doc__ = utils.SaveLoad.save.__doc__

    @classmethod
    def load(cls, *args, **kwargs):
        model = super(UncertainWord2Vec, cls).load(*args, **kwargs)
        # update older models
        if hasattr(model, 'table'):
            delattr(model, 'table')  # discard in favor of cum_table
        if model.negative and hasattr(model.wv, 'index2word'):
            model.make_cum_table()  # rebuild cum_table from vocabulary
        if not hasattr(model, 'corpus_count'):
            model.corpus_count = None
        for v in model.wv.vocab.values():
            if hasattr(v, 'sample_int'):
                break  # already 0.12.0+ style int probabilities
            elif hasattr(v, 'sample_probability'):
                v.sample_int = int(round(v.sample_probability * 2**32))
                del v.sample_probability
        if not hasattr(model, 'syn0_lockf') and hasattr(model, 'syn0'):
            model.syn0_lockf = ones(len(model.wv.syn0), dtype=REAL)
        if not hasattr(model, 'random'):
            model.random = random.RandomState(model.seed)
        if not hasattr(model, 'train_count'):
            model.train_count = 0
            model.total_train_time = 0
        return model

    def _load_specials(self, *args, **kwargs):
        super(UncertainWord2Vec, self)._load_specials(*args, **kwargs)
        # loading from a pre-KeyedVectors word2vec model
        if not hasattr(self, 'wv'):
            wv = KeyedVectors()
            wv.syn0 = self.__dict__.get('syn0', [])
            wv.synDEC = self.__dict__.get('DEC', [])
            wv.syn0norm = self.__dict__.get('syn0norm', None)
            wv.vocab = self.__dict__.get('vocab', {})
            wv.index2word = self.__dict__.get('index2word', [])
            self.wv = wv

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                         limit=None, datatype=REAL):
        """Deprecated. Use gensim.models.KeyedVectors.load_word2vec_format instead."""
        raise DeprecationWarning("Deprecated. Use gensim.models.KeyedVectors.load_word2vec_format instead.")

    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """Deprecated. Use model.wv.save_word2vec_format instead."""
        raise DeprecationWarning("Deprecated. Use model.wv.save_word2vec_format instead.")

    def most_similar(self, positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None):
        return self.wv.most_similar(positive, negative, topn, restrict_vocab, indexer)

    def wmdistance(self, document1, document2):
        return self.wv.wmdistance(document1, document2)

    def most_similar_cosmul(self, positive=[], negative=[], topn=10):
        return self.wv.most_similar_cosmul(positive, negative, topn)

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        return self.wv.similar_by_word(word, topn, restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        return self.wv.similar_by_vector(vector, topn, restrict_vocab)

    def doesnt_match(self, words):
        return self.wv.doesnt_match(words)

    def __getitem__(self, words):
        return self.wv.__getitem__(words)

    def __contains__(self, word):
        return self.wv.__contains__(word)

    def similarity(self, w1, w2):
        return self.wv.similarity(w1, w2)

    def n_similarity(self, ws1, ws2):
        return self.wv.n_similarity(ws1, ws2)

    def predict_output_word(self, context_words_list, topn=10):
        """Report the probability distribution of the center word given the context words as input to the trained model."""
        if not self.negative:
            raise RuntimeError("We have currently only implemented predict_output_word "
                "for the negative sampling scheme, so you need to have "
                "run word2vec with negative > 0 for this to work.")

        if not hasattr(self.wv, 'syn0') or not hasattr(self, 'syn1neg'):
            raise RuntimeError("Parameters required for predicting the output words not found.")

        word_vocabs = [self.wv.vocab[w] for w in context_words_list if w in self.wv.vocab]
        if not word_vocabs:
            warnings.warn("All the input context words are out-of-vocabulary for the current model.")
            return None

        word2_indices = [word.index for word in word_vocabs]

        l1 = np_sum(self.wv.syn0[word2_indices], axis=0)
        if word2_indices and self.cbow_mean:
            l1 /= len(word2_indices)

        prob_values = exp(dot(l1, self.syn1neg.T))     # propagate hidden -> output and take softmax to get probabilities
        prob_values /= sum(prob_values)
        top_indices = matutils.argsort(prob_values, topn=topn, reverse=True)
        return [(self.wv.index2word[index1], prob_values[index1]) for index1 in top_indices]   #returning the most probable output words with their probabilities

    @staticmethod
    def log_accuracy(section):
        return KeyedVectors.log_accuracy(section)

    def accuracy(self, questions, restrict_vocab=30000, most_similar=None, case_insensitive=True):
        most_similar = most_similar or KeyedVectors.most_similar
        return self.wv.accuracy(questions, restrict_vocab, most_similar, case_insensitive)

    @staticmethod
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        return KeyedVectors.log_evaluate_word_pairs(pearson, spearman, oov, pairs)

    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000, case_insensitive=True, dummy4unknown=False):
        return self.wv.evaluate_word_pairs(pairs, delimiter, restrict_vocab, case_insensitive, dummy4unknown)

    def __str__(self):
        return "%s(vocab=%s, size=%s, alpha=%s)" % (self.__class__.__name__, len(self.wv.index2word), self.vector_size, self.alpha)

    def _minimize_model(self, save_syn1 = False, save_syn1neg = False, save_syn0_lockf = False):
        warnings.warn("This method would be deprecated in the future. Keep just_word_vectors = model.wv to retain just the KeyedVectors instance for read-only querying of word vectors.")
        if save_syn1 and save_syn1neg and save_syn0_lockf:
            return
        if hasattr(self, 'syn1') and not save_syn1:
            del self.syn1
        if hasattr(self, 'syn1neg') and not save_syn1neg:
            del self.syn1neg
        if hasattr(self, 'syn0_lockf') and not save_syn0_lockf:
            del self.syn0_lockf
        self.model_trimmed_post_training = True

    def delete_temporary_training_data(self, replace_word_vectors_with_normalized=False):
        """
        Discard parameters that are used in training and score. Use if you're sure you're done training a model.
        If `replace_word_vectors_with_normalized` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!
        """
        if replace_word_vectors_with_normalized:
            self.init_sims(replace=True)
        self._minimize_model()

class DataLoader(object):
    """
    The class for loading uncertain data.
    """

    def __init__(self, domain, files, mode, num_train, beam_size, dataIndexes=[], resample=False, distr_sz=3):
        """
        :param files: The path of the data file.
        """
        self.domain = domain
        self.data = []
        self.files = files
        self.numTrain = num_train
        self.mode = mode
        self.beamSize = beam_size
        self.dataIndexes_ = dataIndexes
        self.resample = resample
        self.action_freq = {}
        self.ground_id = np.inf
        # self.certain_id = -2
        self.distr_sz = distr_sz
        for id, f in enumerate(self.files):
            if 'G_' in f:
                self.ground_id = id
            elif 'N_' in f:
                self.certain_id = id
            else: pass
            if f is None:
                # yield None
                break
            else:
                self.data.append(self.load_data(f))
        # self.useBeamSearch = (self.beamSize == None and [False] or [True])[0]
        #self.useBeamSearch = True and (self.beamSize == None)
        # for f in files:
        #     self.data.append(self.load_data(f))

    def __iter__(self):

        self.uncertain_data = [d for i,d in enumerate(self.data) if i != self.ground_id]
        # self.uncertain_data = data
        self.uncertain_data = sorted(self.uncertain_data, key=lambda x: x[0][0][1], reverse=True)

        if self.mode == 'model1':
            for i in xrange(len(self.data[0])):
                if self.dataIndexes_ == []:
                    PL_RE = self.beam_search_paths([self.uncertain_data[j][i] for j in range(self.distr_sz)], self.beamSize)

                    for p_r in PL_RE:
                        yield p_r
                    continue
                else:
                    if np.any(self.dataIndexes_ == i):
                        PL_RE = self.beam_search_paths([self.uncertain_data[j][i] for j in range(self.distr_sz)], self.beamSize)
                        if self.resample == True:
                            PL_RE = self.resample_data(PL_RE, len(PL_RE))
                        for p_r in PL_RE:
                            yield p_r
                        continue

        elif self.mode == 'model2':
            # for i in xrange(len(self.data[0])):
                # assert len(self.data[0][i]) == len(self.data[1][i]) == len(
                #     self.data[2][i]), "length p0 is %d, p1 is %d, and p2 is %d" % (
                # len(self.data[0][i]), len(self.data[1][i]), len(self.data[2][i]))
                # assert len(self.uncertain_data[0][i])==len(self.uncertain_data[1][i])==len(self.uncertain_data[2][i])==len(self.data[self.certain_id][i]), "length p0 is %d, p1 is %d, p2 is %d, and pG is %d" % (len(self.uncertain_data[0][i]), len(self.uncertain_data[0][i]), len(self.uncertain_data[0][i]), len(self.data[self.certain_id][i]))

                # if i > 1: assert len(self.data[0][i]) == len(self.data[0][i - 1]), (len(self.data[0][i]), len(self.data[0][i - 1]))

            # for i,(p1,p2,p3) in enumerate(zip(self.uncertain_data[0],self.uncertain_data[1],self.uncertain_data[2])):
            # for i, p in enumerate(zip([self.uncertain_data[j] for j in range(self.distr_sz)])):
            # for i, (p1, p2, p3) in enumerate(zip(self.data[0], self.data[1], self.data[2])): # Bug?

            for i in xrange(len(self.uncertain_data[0])):
                p = [self.uncertain_data[j][i] for j in range(self.distr_sz)]
                if self.dataIndexes_ == []:
                    # yield zip(p[0], p[1], p[2])
                    yield zip(*p)
                elif np.any(self.dataIndexes_ == i):
                    # yield zip(p[0], p[1], p[2])
                    yield zip(*p)


            # for i,(p1,p2,p3) in enumerate(zip(self.uncertain_data[0],self.uncertain_data[1],self.uncertain_data[2])):
            #     if self.dataIndexes_ == []:
            #         yield zip(p1,p2,p3)
            #     elif np.any(self.dataIndexes_ == i):
            #         yield zip(p1,p2,p3)

        elif self.mode == 'certain-observation':
            # for i, p in enumerate(self.data[self.certain_id]):
            for i, p in enumerate(self.uncertain_data[0]):
                # assert len(self.uncertain_data[0][i])==len(self.uncertain_data[1][i])==len(self.uncertain_data[2][i])==len(self.data[self.ground_id][i]), "length p0 is %d, p1 is %d, p2 is %d, and pG is %d" % (len(self.uncertain_data[0][i]), len(self.uncertain_data[0][i]), len(self.uncertain_data[0][i]), len(self.data[self.certain_id][i]))
                # c = zip(*p)[0]
                # if 'cut_lettuce_post' in c:
                #     print c
                if self.dataIndexes_ == []:
                    zipP = zip(*p)
                    yield zipP[0]
                else:
                    if np.any(self.dataIndexes_ == i):
                        # zipP = zip(*p)
                        # yield zipP[0]
                        yield p

        elif self.mode == 'ground-truth':
            for i, p in enumerate(self.data[self.ground_id]):
                # assert len(self.uncertain_data[0][i])==len(self.uncertain_data[1][i])==len(self.uncertain_data[2][i])==len(self.data[self.ground_id][i]), "length p0 is %d, p1 is %d, p2 is %d, and pG is %d" % (len(self.uncertain_data[0][i]), len(self.uncertain_data[0][i]), len(self.uncertain_data[0][i]), len(self.data[self.certain_id][i]))
                if self.dataIndexes_ == []:
                    zipP = zip(*p)
                    yield zipP[0]
                else:
                    if np.any(self.dataIndexes_ == i):
                        # zipP = zip(*p)
                        # yield zipP[0]
                        yield p

    def load_data(self, filename):
        if filename is None:
            return None
        UncertainData = []
        with h5py.File(filename, 'r') as h5file:
            for id, data in enumerate(h5file['UncertainData'][:self.numTrain]):
                # data = ast.literal_eval(np.array_repr(data).replace('\n', '')) # https://stackoverflow.com/questions/40719816/extract-list-or-tuple-inside-string
                # data = np.array_repr(data).replace('\n', '')
                if self.domain == "video_salads":
                    data = ast.literal_eval(data) # Uncomment if using the full length TODO fix
                # data = list(map(tuple, data))
                UncertainData.append(data)
        return UncertainData

    def get_action_freq(self, data_ids):
        """
        Get useful information of the dataset loaded.
        :return: Frequency statistics in the form of {word: frequency}
        """
        # num_diff_words

        for id in data_ids:
            if self.data[id] is None:
                continue
            for plan in self.data[id]:
                for action in plan:
                    if action[0] in self.action_freq.keys():
                        self.action_freq[action[0]] += 1
                    else:
                        self.action_freq[action[0]] = 1

    def get_sequence_PER(self, data_ids):
        per_ls = []
        data_mapid = []
        for plan in xrange(len(self.data[0])):
            plan_len = len(self.data[0][plan])
            corr_cnt = 0
            for j in xrange(len(self.data[0][plan])):
                # Get action with highest prob
                a_p = [self.data[d][plan][j] for d in range(len(self.data)) if d != self.ground_id]
                max_a_p = max(a_p, key=lambda x:x[1])
                if max_a_p[0] == self.data[self.ground_id][plan][j][0]:
                    corr_cnt += 1
            per_ls.append(np.true_divide(plan_len - corr_cnt, plan_len))


        # if 3 in data_ids:
        #     data_mapid.append(self.ground_id)
        # if 2 in data_ids:
        #     data_mapid.append(self.certain_id)
        # if len(self.action_freq) == 0:
        #     self.get_action_freq([self.ground_id])
        # action_vocab = self.action_freq.keys()
        # for id in data_mapid:
        #     if self.data[id] is None:
        #         continue
        #     for plan in self.data[id]:
        #         plan_len = len(plan)
        #         cnt = 0
        #         for action in plan:
        #             if action[0] in action_vocab:
        #                 cnt += 1
        #         per_ls.append(np.true_divide(plan_len-cnt,plan_len))
        return np.around(np.mean(per_ls), decimals=4)

    def get_sequence_PER_0727(self, data_ids):
        per_ls = []
        data_mapid = []

        if 3 in data_ids:
            data_mapid.append(self.ground_id)
        if 2 in data_ids:
            data_mapid.append(self.certain_id)
        if len(self.action_freq) == 0:
            self.get_action_freq([self.ground_id])
        action_vocab = self.action_freq.keys()
        for id in data_mapid:
            if self.data[id] is None:
                continue
            for plan in self.data[id]:
                plan_len = len(plan)
                cnt = 0
                for action in plan:
                    if action[0] in action_vocab:
                        cnt += 1
                per_ls.append(np.true_divide(plan_len-cnt,plan_len))
        return np.around(np.mean(per_ls), decimals=4)

    def search_paths(self, data):
        # visited_paths = [[data[0][0]],[data[1][0]]]
        visited_paths = [[data[i][0]] for i in range(len(data))]
        for i in range(1,len(data[0])):
            new_visited_paths = []
            for j in range(len(data)):
                for p in visited_paths:
                    q = deepcopy(p)
                    q.append(data[j][i])
                    new_visited_paths.append(q)
            visited_paths = deepcopy(new_visited_paths)
            gc.collect()

        reliabilities = []
        for path in visited_paths:
            rv = [x[1] for x in path]
            reliabilities.append(np.prod(np.around(np.array(rv, dtype=np.float), 2)))

        # reliabilities = [reduce(lambda x,y: float(x[1])*float(y[1]), path) for path in visited_paths]
        # print len(reliabilities)
        nn = list(zip(reliabilities, visited_paths))
        return nn
        # return visited_paths, reliabilities

    def new_search_paths(self, data):
        # visited_paths = [[data[0][0]],[data[1][0]]]
        visited_paths = [[data[i][0]] for i in range(len(data))]
        for i in range(1,len(data[0])):
            new_visited_paths = []
            for j in range(len(data)):
                for p in visited_paths:
                    q = deepcopy(p)
                    q.append(data[j][i])
                    new_visited_paths.append(q)
            visited_paths = deepcopy(new_visited_paths)
            gc.collect()

        reliabilities = []
        for path in visited_paths:
            rv = [x[1] for x in path]
            reliabilities.append(np.prod(np.around(np.array(rv, dtype=np.float), 2)))

        # reliabilities = [reduce(lambda x,y: float(x[1])*float(y[1]), path) for path in visited_paths]
        # print len(reliabilities)
        nn = list(zip(reliabilities, visited_paths))
        return nn
        # return visited_paths, reliabilities

    def beam_search_paths(self, data, beam_size):
        # Init: Beam size, root nodes, a SET for storing paths
        # PQ = PriorityQueue(100)
        # visited_paths = []
        visited_paths = [(float(data[i][0][1]), [data[i][0]]) for i in range(len(data))]
        # heapq.heapify(visited_paths)
        # for i in range(len(data)):
        #     # print data[i][0][1]
        #     # print (float(data[i][0][1]), data[i][0])
        #     heapq.heappush(visited_paths, (float(data[i][0][1]), [data[i][0]]))

        # Grow path-reliability pairs -> SET
        for i in range(1,len(data[0])):
            new_visited_paths = []
            for j in range(len(data)):
                for p in visited_paths:
                    r_p = p[0]
                    q = deepcopy(p[1])
                    q.append(data[j][i])
                    r_q = r_p * float(data[j][i][1])

                    # If size(SET) >= size(Beam): Delete the path with lowest reliability in SET
                    if len(new_visited_paths) < beam_size:
                        heapq.heappush(new_visited_paths, (r_q, q))
                    else: heapq.heappushpop(new_visited_paths, (r_q, q))

                    # new_visited_paths.append((r_q, q))
            visited_paths = deepcopy(new_visited_paths)

            # if beam_size - len(visited_paths) < 0:
            #     visited_paths = heapq.nlargest(beam_size, new_visited_paths)
            # else: visited_paths = heapq.heapify(new_visited_paths)

        # return visited_paths
        nn = list(visited_paths)
        return nn

    # def resample_data(self, data, beam_size):
    #     dict_data = {i:list(data[i]) for i in range(beam_size)}
    #     weights = zip(*list(data))[0]
    #     norm_weights = [float(i)/sum(weights) for i in weights]
    #     particles = dict_data.keys()
    #     new_particles = self.roulette_resample(particles, norm_weights, beam_size)
    #     new_data = [dict_data[i] for i in new_particles]
    #     return new_data

    def resample_data(self, data, beam_size):
        # import time
        # if (itr*100)/len(list_of_actions) % 10 == 0:
        #     sys.stdout.write( "\rProgress: %s %%" % str( (itr*100)/len(list_of_actions) ) )
        #     sys.stdout.flush()
        # print "beam_size", beam_size

        weights = np.array(zip(*list(data))[0])
        # norm_weights = [float(i)/sum(weights) for i in weights]
        norm_weights = weights / np.linalg.norm(weights)
        # norm_weights = normalize(weights[:, np.newaxis], axis=0).ravel()
        particles = np.arange(beam_size)
        # curr_time = time.time()
        new_particles = self.roulette_resample(particles, norm_weights, beam_size)
        # print "resampling time is %.6f" % (time.time()-curr_time)

        new_data = [data[i] for i in new_particles]
        return new_data

    def roulette_resample(self, particles, weight, N):
        # Resamples particles. Particles with higher weight have higher probability
        # of surviving.
        #
        # Uses roulette wheel algorithm for resambling.

        # if N > 10000:
        #     N = 10000
        sampledParticles = []
        index = int(random.random() * N)
        beta = 0.0
        maxWeight = max(weight)
        for i in xrange(N):
            beta += random.random() * 2.0 * maxWeight
            while beta > weight[index]:
                beta -= weight[index]
                index = (index + 1) % N
            sampledParticles.append(particles[index])
            # heapq.heappush(sampledParticles, particles[index])
        return sampledParticles


if __name__=='__main__':
    file1 = "data1_5.h5"
    file2 = "data2_5.h5"
    file3 = "dataG_5.h5"
    files = file1, file2, file3
    obj = DataLoader(files)
    test = [[('u',2),('y',3)],[('x',4),('w',5)]]
    print obj.search_paths(test)
