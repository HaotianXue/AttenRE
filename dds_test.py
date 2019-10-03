import torch
import numpy as np
import data_fetcher
from sklearn.metrics import roc_auc_score, average_precision_score

'''
This file is used to test the dds model
'''

embed_dim = 50
w2v_path = './data/word2vec.txt'
rel_path = './data/nyt/relation2id.txt'
test_path = './data/nyt/test.txt'
# test for cnn dds model
test_fetcher = data_fetcher.NYTFetcher(w2v_path, rel_path, embed_dim, test_path, max_sen_len=100, padding=True)
# test for rnn dds model
rnn_test_fetcher = data_fetcher.NYTFetcher(w2v_path, rel_path, embed_dim, test_path, max_sen_len=100)


def evaluation(prob_y, target_y):
    num_test, num_rel = prob_y.shape
    target_prob = np.reshape(prob_y[:, 1:], (-1))  # note that the relation of the first column is NA
    target_y = np.array(target_y)
    target_y = np.reshape(target_y[:, 1:], (-1))
    ordered_idx = np.argsort(-target_prob)
    print('Total validation count %d' % (np.sum(target_y)))
    top_n = [100, 200, 300]
    prec_at_n = np.zeros(len(top_n))
    for k, top_k in enumerate(top_n):
        prec_at_n[k] = np.sum(target_y[ordered_idx][:top_k], dtype=float) / float(top_k)
        print("Precision @ %d: %f" % (top_k, prec_at_n[k]))

    roc_auc = roc_auc_score(target_y, target_prob)
    print("ROC-AUC score: %f" % (roc_auc))
    ap = average_precision_score(target_y, target_prob)
    print("Average Precision: %f" % (ap))


def test(test_data, model):
    with torch.no_grad():
        all_y = list()
        all_predicted_y = list()
        for x, y in test_data:
            output = model(x)
            predicted_y = output.data.numpy()
            print('predicted_y: ', predicted_y.shape)
            print(predicted_y)
            print(y)
            break
            _y = torch.from_numpy(y).float()
            all_y.append(y)
            all_predicted_y.append(predicted_y)
        evaluation(np.array(all_predicted_y), np.array(all_y))


print('===> Loading the cnn model')
net = torch.load('./trained_model/dds_cnn_v1.pt')
print('Finished loading the cnn model')

'''
print('===> Loading the rnn model')
net = torch.load('./trained_model/dds_rnn_v1.pt')
print('Finished loading the rnn model')
'''

print('===> Start testing')
test(test_fetcher, net)
print('Finished testing')
