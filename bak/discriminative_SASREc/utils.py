import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    """
    Negative sampling.
    NB: also this function can be modified if we need to change the training of the model.
    """

    # sampler for batch generation
    def random_neq(l, r, s):
        """
        Repetitive sampling until the sample is not in s.
        """
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """
    It is needed for the negative sampling.
    """
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(fname):
    """
    Function to generate the data splitting for train/val/test split.
    This function read the dataset file, and then extract the number of users, and items in the dataset.
    Then it splits the dataset into train/val/test. If a users has less than 4 interactions, it will not be used for the
    validation/test split. Else, the last interaction is used for test, while the second to last is used for training.
    NB: Also, this function is specific for this tipe of discriminative training, potentially to be modified if we want
    to use generative training.
    NB: This function doesn't implement k-core.
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:  # To be rigorous, the training set needs at least two data points to learn
            # if the user has less than 4 interaction, it goes in to the training set only
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # else the test set is the last item in the sequence, while the validations is the second to last.
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate_model(model, dataset, args, mode):
    """
    Unified evaluation function for validation and test sets.
    mode: 'test' or 'valid'
    """

    [train, valid, test, usernum, itemnum] = dataset

    random.seed(args.seed)
    np.random.seed(args.seed)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # Limit to 10k users for speed if dataset is huge
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1:
            continue

        if mode == 'test' and len(test[u]) < 1:
            continue

        if mode == 'valid' and len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)

        if mode == 'test':
            full_history = train[u] + valid[u]
            target_item = test[u][0]
        else:
            full_history = train[u]
            target_item = valid[u][0]

        # padding
        cut = full_history[-args.maxlen:]
        seq[-len(cut):] = cut

        # negative sampling of 100 negative
        rated = set(train[u])
        rated.add(0)

        # valid e test should not be negative (difference from original implementation)
        if len(valid[u]) > 0:
            rated.add(valid[u][0])
        if len(test[u]) > 0:
            rated.add(test[u][0])

        item_idx = [target_item]

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]  # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate(model, dataset, args):
    return evaluate_model(model, dataset, args, mode='test')


def evaluate_valid(model, dataset, args):
    return evaluate_model(model, dataset, args, mode='valid')
