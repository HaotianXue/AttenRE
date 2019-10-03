"""
Data fetcher for supervised models
DataFetcher is an abstract class handling dataset.
Each data point feed into model will consist of
    (sentence, pos1, pos2, relations)
"""
from abc import abstractmethod
import numpy as np
from collections import defaultdict

# process draft
OOV = 'OOV'  # 'out of vocabulary'
BLANK = 'BLANK'
max_sen_len = 100  # Predefined maximum length of sentence, need for position embedding


class DataLoader():
    def __init__(self, w2v_path='', emb_dim='', data_path='', rel_path='', is_shuffle=True, batch_size=4, max_sen_len=100, padding=False):
        self.w2v_path = w2v_path
        self.data_path = data_path
        self.rel_path = rel_path
        self.word2id = {}
        self.is_shuffle = is_shuffle
        self.max_sen_len = max_sen_len
        self.current_pos = 0
        self.emb_dim = emb_dim
        self.word2id, self.word_embedding = self.load_w2v(w2v_path, emb_dim)
        self.num_voca = len(self.word2id)
        self.padding = padding
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    @abstractmethod
    def next(self):
        """This will return next training pair x, y"""

    @abstractmethod
    def next_batch(self):
        """THis will return next batch of training pairs"""

    def reset(self):
        """Reset the generator position to start"""
        self.current_pos = 0

    def load_w2v(self, path, dim):
        vec = []
        word2id = {}
        word2id[BLANK] = len(word2id)
        word2id[OOV] = len(word2id)
        vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
        vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
        with open(path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                word2id[tokens[0]] = len(word2id)
                vec.append(np.array([float(x) for x in tokens[1:]]))

        word_embeddings = np.array(vec, dtype=np.float32)
        return word2id, word_embeddings


# TODO: 写一个semEval dataset的fetcher用作supervised relation classification的任务上
class SemEvalFetcher(DataLoader):
    # (sen, pos_info, relation)

    def __init__(self, *args, **kwargs):
        super(SemEvalFetcher, self).__init__(*args, **kwargs)
        self.labelsMapping = {'Other': 0,
                              'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                              'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                              'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                              'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                              'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                              'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                              'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                              'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                              'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
        self.num_samples, self.data_x, self.data_y, self.seq_len = self._load_relations(self.data_path)
        self.indices = np.arange(0, self.num_samples)
        if self.is_shuffle:
            np.random.shuffle(self.indices)

    def _load_relations(self, path):
        import nltk
        data_x = []
        data_y = []
        lines = [line.strip() for line in open(path)]
        num_samples = 0
        seq_len = []
        for idx in range(0, len(lines), 4):
            sent2id = list()
            relation = lines[idx + 1]

            sentence = lines[idx].split("\t")[1][1:-1]
            sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
            sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")

            tokens = nltk.word_tokenize(sentence)

            tokens.remove('_/e1_')
            tokens.remove('_/e2_')

            e1 = tokens.index("_e1_")
            del tokens[e1]
            e2 = tokens.index("_e2_")
            del tokens[e2]

            for token in tokens:
                if token in self.word2id:
                    sent2id.append(self.word2id[token])
                else:
                    sent2id.append(self.word2id[OOV])
            if len(tokens) < self.max_sen_len:
                num_samples += 1
                sen_len = len(sent2id)
                seq_len.append(sen_len)
                sent2id = np.array(sent2id)
                pos1vec = np.arange(self.max_sen_len - e1, self.max_sen_len - e1 + sen_len)
                pos2vec = np.arange(self.max_sen_len - e2, self.max_sen_len - e2 + sen_len)
                if self.padding:
                    sent2id = self.pad(sent2id, self.max_sen_len)
                    pos1vec = self.pad(pos1vec, self.max_sen_len)
                    pos2vec = self.pad(pos2vec, self.max_sen_len)

                data_x.append([sent2id, pos1vec, pos2vec])
                data_y.append(self.labelsMapping[relation])

        return num_samples, data_x, np.array(data_y), seq_len

    def next(self):
        while True:
            if self.current_pos >= self.num_samples:
                raise StopIteration
            index = self.indices[self.current_pos]
            x = self.data_x[index]  #
            y = self.data_y[index]  # relation2id
            sen_len = self.seq_len[index]  # 这句句子的实际长度
            self.current_pos += 1
            '''
            sent2id, en1pos, en2pos, relation2id = triple
            sen_len = len(sent2id)
            pos1vec = np.arange(self.max_sen_len - en1pos, self.max_sen_len - en1pos + sen_len)
            pos2vec = np.arange(self.max_sen_len - en2pos, self.max_sen_len - en2pos + sen_len)
            if self.padding:
                sent2id = self.pad(sent2id, self.max_sen_len)
                pos1vec = self.pad(pos1vec, self.max_sen_len)
                pos2vec = self.pad(pos2vec, self.max_sen_len)
            x = (np.array(sent2id), pos1vec, pos2vec)
            y = relation2id
            '''
            return x, y, sen_len

    def next_batch(self):
        batch_x = list()
        batch_y = list()
        seq_lens = list()
        while True:
            if self.current_pos >= self.num_samples:
                raise StopIteration
            training_pair = self.next()
            batch_x.append(training_pair[0])
            batch_y.append(training_pair[1])
            seq_lens.append(training_pair[2])
            if self.current_pos % self.batch_size == 0 or self.current_pos == self.num_samples:
                # print('batch size: ', len(batch_y), ' pos: ', self.current_pos, ' num of samples: ', self.num_samples)
                if self.padding:  # return a form of matrix if padding
                    batch_x = np.stack(batch_x, axis=0)
                return batch_x, np.array(batch_y), np.array(seq_lens)

    def reset(self):
        super(SemEvalFetcher, self).reset()
        np.random.shuffle(self.indices)

    def pad(self, sen, max_sen_len):
        '''
        为了cnn中句子长度一致(rnn可以不用)
        sen: numpy array
        :return: numpy array
        '''
        padding = np.zeros((max_sen_len - sen.shape[0]), dtype=int)
        return np.hstack((sen, padding))


# TODO: 写一个NYT的dataloader
class NYTFetcher(DataLoader):

    def __init__(self, *args, **kwargs):
        super(NYTFetcher, self).__init__(*args, **kwargs)
        self.rel2id = {}
        self.rel2id, self.id2rel = self.extract_relations(self.rel_path)
        self.num_rel = len(self.rel2id)
        self.num_samples, self.data_x, self.data_y, self.seq_len = self._load_relations(self.data_path)
        self.indices = np.arange(0, self.num_samples)
        if self.is_shuffle:
            np.random.shuffle(self.indices)

    def _load_relations(self, file_path):
        # data_x: (sen, pos1, pos2)
        # data_y: (relation2id)
        data_x = []
        data_y = []
        seq_len = []
        num_samples = 0
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.split('\t')
                # the number of tokens are not the same for train.txt and test.txt, cannot unpack directly from split
                en1id, en2id, en1token, en2token, rel, sen = tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], \
                                                             tokens[5]
                data_y.append(self.rel2id[rel] if rel in self.rel2id.keys() else 0)

                tokens = sen.split()
                if len(tokens) < self.max_sen_len:
                    num_samples += 1
                    sent2id = list()
                    en1pos, en2pos = -1, -1
                    for pos, token in enumerate(tokens):
                        if token == en1token:
                            en1pos = pos
                        if token == en2token:
                            en2pos = pos

                        if token in self.word2id:
                            sent2id.append(self.word2id[token])
                        else:
                            sent2id.append(self.word2id[OOV])
                    sen_len = len(sent2id)
                    seq_len.append(sen_len)
                    sent2id = np.array(sent2id)
                    pos1vec = np.arange(self.max_sen_len - en1pos, self.max_sen_len - en1pos + sen_len)
                    pos2vec = np.arange(self.max_sen_len - en2pos, self.max_sen_len - en2pos + sen_len)
                    if self.padding:
                        sent2id = self.pad(sent2id, self.max_sen_len)
                        pos1vec = self.pad(pos1vec, self.max_sen_len)
                        pos2vec = self.pad(pos2vec, self.max_sen_len)
                    data_x.append([sent2id, pos1vec, pos2vec])
        return num_samples, data_x, data_y, seq_len

    def extract_relations(self, rel_path):
        rel2id = dict()
        id2rel = dict()

        with open(rel_path, 'r') as f:
            for line in f:
                rel, id = line.strip().split()
                rel2id[rel] = int(id)
                id2rel[int(id)] = rel
        return rel2id, id2rel

    def next(self):
        while True:
            if self.current_pos >= self.num_samples:
                raise StopIteration
            index = self.indices[self.current_pos]
            x = self.data_x[index]  # [(sent2id, pos1vec, pos2vec)]
            y = self.data_y[index]  # relation2id
            sen_len = self.seq_len[index]
            self.current_pos += 1
            return x, y, sen_len

    def next_batch(self):
        batch_x = list()
        batch_y = list()
        seq_lens = list()
        while True:
            if self.current_pos >= self.num_samples:
                raise StopIteration
            training_pair = self.next()
            batch_x.append(training_pair[0])
            batch_y.append(training_pair[1])
            seq_lens.append(training_pair[2])
            if self.current_pos % self.batch_size == 0 or self.current_pos == self.num_samples:
                if self.padding:  # return a form of matrix if padding
                    batch_x = np.stack(batch_x, axis=0)
                return batch_x, np.array(batch_y), np.array(seq_lens)

    def reset(self):
        super(NYTFetcher, self).reset()
        np.random.shuffle(self.indices)

    def pad(self, sen, max_sen_len):
        '''
        为了cnn中句子长度一致(rnn可以不用)
        sen: numpy array
        :return: numpy array
        '''
        padding = np.zeros((max_sen_len - sen.shape[0]), dtype=int)
        return np.hstack((sen, padding))


# TODO: dds版的NYT fetcher
class ddsNYTFetcher(DataLoader):
    def __init__(self, *args, **kwargs):
        super(ddsNYTFetcher, self).__init__(*args, **kwargs)
        self.rel2id, self.id2rel = self._load_relations(self.rel_path)
        self.num_rel = len(self.rel2id)
        self.pairs, self.sen_col = self._load_triple(self.data_path)
        self.keys = list(self.pairs.keys())  # keys of triples-dictionary, key=(en1id, en2id), value=(rel1, rel2, ...)
        # self.num_pairs = len(self.pairs)
        self.num_pairs = len(self.sen_col)
        if self.is_shuffle:
            np.random.shuffle(self.keys)

    def _load_relations(self, path):
        rel2id = dict()
        id2rel = dict()

        with open(path, 'r') as f:
            for line in f:
                rel, id = line.strip().split()
                rel2id[rel] = int(id)
                id2rel[int(id)] = rel
        return rel2id, id2rel

    def _load_triple(self, datapath):
        """
        extract knowledge graph and sentences from nyt dataset
        :return: - triples: dictionary, key=(en1id, en2id), value=(rel1, rel2, ...)
                 - sen_col: dictionary, key=(en1id, en2id), value=((sen2tid :: list, en1pos :: int, en2pos :: int), ...)
        """
        triples = defaultdict(set)
        sen_col = defaultdict(list)
        with open(datapath, 'r') as f:
            for line in f:
                tokens = line.split('\t')
                # the number of tokens are not the same for train.txt and test.txt, cannot unpack directly from split
                en1id, en2id, en1token, en2token, rel, sen = tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], \
                                                             tokens[5]

                # add relation
                triples[(en1id, en2id)].add(rel)

                # find entity positions
                tokens = sen.split()
                if len(tokens) < self.max_sen_len:
                    sen2tid = list()
                    en1pos, en2pos = -1, -1
                    for pos, token in enumerate(tokens):
                        if token == en1token:
                            en1pos = pos
                        if token == en2token:
                            en2pos = pos

                        if token in self.word2id:
                            sen2tid.append(self.word2id[token])
                        else:
                            sen2tid.append(self.word2id[OOV])

                    # add parsed sentence
                    sen_col[(en1id, en2id)].append((sen2tid, en1pos, en2pos))

        for key, value in triples.items():
            # if entity pair has any relation except 'NA', remove 'NA' from relation set
            if len(value) > 1:
                if 'NA' in value:
                    value.remove('NA')

        return triples, sen_col

    def next(self):
        """
        Generate batch of sentences from a randomly sampled entity pair
        :param triples: dictionary, key=(en1id, en2id), value=(rel1, rel2, ...)
        :param sen_col: dictionary, key=(en1id, en2id), value=((sen2tid, en1pos, en2pos), ...)
        :param rel2id: dictionary, key=relation, value=id
        :return:
        """
        while True:
            if self.current_pos >= self.num_pairs:
                raise StopIteration

            key = self.keys[self.current_pos]
            self.current_pos += 1
            x = list()
            y = np.zeros(self.num_rel)
            for rel in self.pairs[key]:  # all relations of given key(entities <=> (en1id, en2id))
                if rel in self.rel2id:
                    y[self.rel2id[rel]] = 1
                else:
                    y[0] = 1  # 0 in rel2id.keys() is NA
            sentences = self.sen_col[key]  # all sentences that contain given entities
            seq_len = list()
            for sen2tid, en1pos, en2pos in sentences:
                sen_len = len(sen2tid)
                seq_len.append(sen_len)
                sen2tid = np.array(sen2tid)
                pos1vec = np.arange(self.max_sen_len - en1pos, self.max_sen_len - en1pos + sen_len)
                pos2vec = np.arange(self.max_sen_len - en2pos, self.max_sen_len - en2pos + sen_len)
                if self.padding:  # TODO: [(sen2id, pos1, pos2)] => (batch, 3, max_sen_len)
                    sen2tid = self.pad(sen2tid, self.max_sen_len)  # (max_sen_len)
                    pos1vec = self.pad(pos1vec, self.max_sen_len)
                    pos2vec = self.pad(pos2vec, self.max_sen_len)
                    stacked = np.stack((sen2tid, pos1vec, pos2vec), axis=0)
                    x.append(stacked)  # [(3, max_sen_len)]
                else:
                    x.append((sen2tid, pos1vec, pos2vec))

            if len(x) > 0:
                if self.padding:
                    x = np.stack(x, axis=0)  # (batch, 3, max_sen_len)
                return x, y, np.array(seq_len)

    def next_batch(self):
        return self.next()

    def reset(self):
        super(ddsNYTFetcher, self).reset()
        np.random.shuffle(self.keys)

    def pad(self, sen, max_sen_len):
        '''
        为了cnn中句子长度一致
        sen: numpy array
        :return: numpy array
        '''
        padding = np.zeros((max_sen_len - sen.shape[0]), dtype=int)
        return np.hstack((sen, padding))
