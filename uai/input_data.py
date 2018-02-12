import os
import random
import numpy as np

class DataSet(object):
    def __init__(self, head_list, tail_list, relation_list,
                 head_set, tail_set, entity_total, tag_total):
        self._num_examples = len(head_list)
        self._head_list = np.array(head_list)
        self._tail_list = np.array(tail_list)
        self._relation_list = np.array(relation_list)
        self._head_set = np.array(head_set)
        self._tail_set = np.array(tail_set)
        self._entity_total = entity_total
        self._tag_total = tag_total
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def entity_total(self):
        return self._entity_total
    @property
    def tag_total(self):
        return self._tag_total
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, aeBeta):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
	    perm = np.random.permutation(np.arange(self._num_examples))
	    perm = perm.flatten().tolist()
            self._head_list = self._head_list[perm]
            self._tail_list = self._tail_list[perm]
            self._relation_list = self._relation_list[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        
        tag_size = self._tag_total
        entity_size = self._entity_total
        pos_h = []
        pos_t = []
        pos_r = []
        pos_b = []
        neg_h = []
        neg_t = []
        neg_r = []
        neg_b = []
        for i in range(start, end):
            cur_h = self._head_list[i]
            cur_t = self._tail_list[i]
            cur_r = self._relation_list[i]
            set_r = set(cur_r)
            r_one_hot = np.zeros(tag_size, dtype=float)
            b = np.ones(tag_size, dtype=float)
            r_one_hot[cur_r] = 1.0
            b[cur_r] = aeBeta
            #replace head
            pos_h.append(cur_h)
            pos_t.append(cur_t)
            pos_r.append(r_one_hot)
            pos_b.append(b)
            rand_h = random.randint(0, entity_size-1)
            while(rand_h in self._tail_set[cur_t]):
                rand_h = random.randint(0, entity_size-1)
            neg_h.append(rand_h)
            neg_t.append(cur_t)
            neg_r.append(r_one_hot)
            neg_b.append(b)
            #repalce tail
            pos_h.append(cur_h)
            pos_t.append(cur_t)
            pos_r.append(r_one_hot)
            pos_b.append(b)
            rand_t = random.randint(0, entity_size-1)
            while(rand_t in self._head_set[cur_h]):
                rand_t = random.randint(0, entity_size-1)
            neg_h.append(cur_h)
            neg_t.append(rand_t)
            neg_r.append(r_one_hot)
            neg_b.append(b)
            #replace relation
            pos_h.append(cur_h)
            pos_t.append(cur_t)
            pos_r.append(r_one_hot)
            pos_b.append(b)
            rand_set_r = set([])
            rand_r = random.randint(0, tag_size-1)
            len_r = len(cur_r)
            while(len(rand_set_r) < len_r and len(rand_set_r) + len_r < tag_size):
                if (rand_r not in set_r) and (rand_r not in rand_set_r):
                    rand_set_r.add(rand_r)
                rand_r = random.randint(0, tag_size-1)
            rand_cur_r = [r for r in rand_set_r]
            rand_r_one_hot = np.zeros(tag_size, dtype=float)
            rand_r_one_hot[rand_cur_r] = 1.0
            rand_b = np.ones(tag_size, dtype=float)
            rand_b[rand_cur_r] = aeBeta
            neg_h.append(cur_h)
            neg_t.append(cur_t)
            neg_r.append(rand_r_one_hot)
            neg_b.append(rand_b)
        return np.array(pos_h), np.array(pos_t), np.array(pos_r), np.array(pos_b),\
               np.array(neg_h), np.array(neg_t), np.array(neg_r), np.array(neg_b)

    def next_test_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        
        tag_size = self._tag_total
        entity_size = self._entity_total
        pos_h = []
        pos_t = []
        pos_r = []
        for i in range(start, end):
            cur_h = self._head_list[i]
            cur_t = self._tail_list[i]
            cur_r = self._relation_list[i]
            r_one_hot = np.zeros(tag_size, dtype=float)
            b = np.ones(tag_size, dtype=float)
            r_one_hot[cur_r] = 1.0
            pos_h.append(cur_h)
            pos_t.append(cur_t)
            pos_r.append(r_one_hot)
        return np.array(pos_h), np.array(pos_t), np.array(pos_r)

    def next_autoencoder_batch(self, batch_size, aeBeta):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
	    perm = np.random.permutation(np.arange(self._num_examples))
	    perm = perm.flatten().tolist()
            self._head_list = self._head_list[perm]
            self._tail_list = self._tail_list[perm]
            self._relation_list = self._relation_list[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        vec_size = self._tag_total
        vecs = []
        bs = []
        for i in range(start, end):
            vec_index = self._relation_list[i]
            vec = np.zeros(vec_size, dtype=float)
            b = np.ones(vec_size, dtype=float)
            vec[vec_index] = 1.0
            b[vec_index] = aeBeta
            vecs.append(vec)
            bs.append(b)
        return np.array(vecs), np.array(bs)

def read_triples(filename):
    head_list = []
    tail_list = []
    relation_list = []
    head_set = []
    tail_set = []
    fin = open(filename, 'r')
    content = fin.readline()
    _, entity_total, tag_total = [int(i) for i in content.split()]
    for i in xrange(entity_total):
        head_set.append(set())
        tail_set.append(set())
    while 1:
        content = fin.readline()
        if content == '':
            break
        values = [int(i) for i in content.split()]
        head_list.append(values[0])
        tail_list.append(values[1])
        head_set[values[0]].add(values[1])
        tail_set[values[1]].add(values[0])
        relation_list.append(values[2:])
    fin.close()
    return DataSet(head_list, tail_list, relation_list,
                   head_set, tail_set, entity_total, tag_total)


def read_data_sets(train_dir='aminer_small'):
    class DataSets(object):
        pass
    data_sets = DataSets()
    TRAIN = 'train.txt'
    VALID = 'valid.txt'
    TEST = 'test.txt'
    data_sets.train = read_triples(os.path.join(train_dir, TRAIN))
    data_sets.valid = read_triples(os.path.join(train_dir, VALID))
    data_sets.test = read_triples(os.path.join(train_dir, TEST))
    data_sets.entity_total = data_sets.train.entity_total
    data_sets.tag_total = data_sets.train.tag_total
    return data_sets
