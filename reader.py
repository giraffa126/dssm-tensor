# coding: utf8
import sys
import os
import collections
import random
import pickle

class DataReader(object):
    def __init__(self, vocab_path, data_path, vocab_size=1000000, batch_size=512, max_seq_len=48):
        """ init
        """
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._max_seq_len = max_seq_len
        if not os.path.exists(vocab_path):
            self._word_to_id = self._build_vocab(data_path)
            with open(vocab_path, "w") as ofs:
                pickle.dump(self._word_to_id, ofs)
        else:
            with open(vocab_path, "r") as ifs:
                self._word_to_id = pickle.load(ifs)
        self._data = self._build_data(data_path)

    def _build_vocab(self, filename):
        with open(filename, "r") as ifs:
            data = ifs.read().replace("\n", " ").replace("\t", " ").split()
        counter = collections.Counter(data)
        count_pairs = counter.most_common(self._vocab_size - 2)

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(2, len(words) + 2)))
        word_to_id["<pad>"] = 0
        word_to_id["<unk>"] = 1
        print("vocab words num: ", len(word_to_id))
        return word_to_id

    def _build_data(self, filename, is_shuffle=True):
        with open(filename, "r") as ifs:
            lines = ifs.readlines()
            data = list(map(lambda x: x.strip().split("\t"), lines))
            random.shuffle(data)
        return data

    def _padding_batch(self, batch):
        for idx, line in enumerate(batch[0]):
            # padding query
            if len(line) > self._max_seq_len:
                batch[0][idx] = line[:self._max_seq_len]
            else:
                batch[0][idx] = line + [self._word_to_id["<pad>"]] * (self._max_seq_len - len(line)) 
        for idx, line in enumerate(batch[1]):
            # padding title
            if len(line) > self._max_seq_len:
                batch[1][idx] = line[:self._max_seq_len]
            else:
                batch[1][idx] = line + [self._word_to_id["<pad>"]] * (self._max_seq_len - len(line)) 
        # neg sample
        for idx, line in enumerate(batch[1]):
            neg_idx = random.randint(0, len(batch[1]) -1)
            while neg_idx == idx:
                neg_idx = random.randint(0, len(batch[1]) -1)
            batch[2].append(batch[1][neg_idx])
        return batch

    def batch_generator(self):
        curr_size = 0
        batch = [[], [], []]
        for line in self._data:
            if len(line) != 2:
                continue
            curr_size += 1
            query, title = line
            query_ids = [self._word_to_id.get(x, self._word_to_id["<unk>"]) for x in query.split()]
            title_ids = [self._word_to_id.get(x, self._word_to_id["<unk>"]) for x in title.split()]
            batch[0].append(query_ids)
            batch[1].append(title_ids)
            if curr_size >= self._batch_size:
                yield self._padding_batch(batch)
                batch = [[], [], []]
                curr_size = 0
        if curr_size > 0:
            yield self._padding_batch(batch)

    def extract_emb_generator(self):
        curr_size = 0
        batch = []
        for line in self._data:
            if len(line) != 1:
                continue
            curr_size += 1
            query = line[0]
            query_ids = [self._word_to_id.get(x, self._word_to_id["<unk>"]) for x in query.split()]
            query_ids = query_ids + [self._word_to_id["<pad>"]] * (self._max_seq_len - len(query_ids))
            batch.append(query_ids)
            if curr_size >= self._batch_size:
                yield batch
                batch = []
                curr_size = 0
        if curr_size > 0:
            yield batch

if __name__ == "__main__":
    reader = DataReader("data/vocab.pkl", "data/query.txt")
    for batch in reader.extract_emb_generator():
        for idx, line in enumerate(batch):
            print line
