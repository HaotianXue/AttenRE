import torch
import numpy as np
import data_loader
from sklearn.metrics import roc_auc_score, average_precision_score


embed_dim = 300
w2v_path = './glove.6B/glove.6B.300d.txt'
sen_path = './data/nyt/test.txt'
rel_path = './data/nyt/relation2id.txt'
print('===> Fetching the data')
test_fetcher = data_loader.ddsNYTFetcher(w2v_path=w2v_path, emb_dim=embed_dim, data_path=sen_path, rel_path=rel_path,
                                         max_sen_len=100, padding=True)
print('Finished fetching the data')


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
            # print('predicted_y: ', predicted_y.shape)
            _y = torch.from_numpy(y).float()
            all_y.append(y)
            all_predicted_y.append(predicted_y)
        evaluation(np.array(all_predicted_y), np.array(all_y))


print('===> Loading the trained model')
model = torch.load('./trained_model/dds_transformer_v1.pt')
print('Finished loading the trained model')

print('===> Start testing')
test(test_fetcher, model)
print('Finished testing')
