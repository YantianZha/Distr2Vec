The main code is U_train_and_test_final_eval.py. The data include a visual grounded plan corpora in /video_salads, and a synthetic plan corpora in /salads
In video_salads, /distr_3_91100 is the plan corpora of 8% PER, and /distr_3_100 of 69% PER
In salads, we have three synthetic plan corpora of sequence length 10, 20 and 30, for PER = 50%

Install environment and dependencies:
1. Install Anaconda
2. Install fundamental dependencies, e.g., python 2.7, numpy
3. Install cntk: conda install -c conda-forge cntk-gpu // or replace cntk-gpu with cntk
4. Add gensim to python path

We have the following arguments:
folders: the name of dataset folder in either salads (synthetic corpora) or video_salads (visual grounded corpora)
win_range: the lower and upper bound of word2vec/distr2vec context window size
eval_hs: if using hierarchical softmax (hs) not. Give 0 or 1 or 2. 0 means only use non-hs, 1 means only use hs, 2 means use non-hs and then hs.
domain: either salads of video_salads
mode: middle_random, means we randomly sample positions to remove observed distributions for each plan
num_missing: the number or percentage of positions with missing observations in a plan. If inputting a float value between 0 and 1, the value will be treated as a percentage. Otherwise if inputting an integer larger than 0, it will be treated as a number of missing actions.
top_k: lower and upper bound of candidate predictions for a step
num_train: number of training instances that the are used from a dataset. If use all please input 0 or nothing (default is 0)
beam_sizes: number of deterministic samples that the RBM should get from each uncertain plan
cv_split: cross validation splits number
iter: number of training epoches for word2vec or distr2vec

To evaluate the visual grounded corpora, run U_train_and_test_final_eval.py with the arguments:
--folders /distr_3_100/ /distr_3_91100/ --win_range 1 1 --eval_hs 2 --domain video_salads --mode middle_random --top_k 3 3 --num_train 0 --beam_sizes 1 3 5  --num_missings 0.1 --cv_split 6 --iter 60

To evaluate the synthetic corpora, run U_train_and_test_final_eval.py with the arguments:
--folders /halff10noisy/ /halff20noisy/ /halff30noisy/ --win_range 1 1 --eval_hs 1 --domain salads --mode middle_random --top_k 3 3 --num_train 0 --beam_sizes 1 3 5  --num_missings 1 --cv_split 6 --iter 60

