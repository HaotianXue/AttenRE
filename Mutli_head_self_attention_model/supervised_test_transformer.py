import torch
import numpy as np
import data_loader
from sklearn.metrics import f1_score


embed_dim = 300
w2v_path = 'glove.6B.300d.txt'
sen_path = '../TEST_FILE_FULL.txt'
print('===> Fetching the data')
test_fetcher = data_loader.SemEvalFetcher(w2v_path=w2v_path, emb_dim=embed_dim, data_path=sen_path,
                                          max_sen_len=100, padding=True)
print('Finished fetching the data')


def test(test_data, model):
    with torch.no_grad():
        correct = 0
        total = 0
        # matrix used for computing f1 score
        y_true = list()
        y_pred = list()
        for i, data in enumerate(test_data):
            inputs, labels, seq_len = data
            labels = torch.from_numpy(labels)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # predicted shape: [batch_size, 1]
            total += labels.size(0)  # labels shape: [batch_size, 1]
            correct += (predicted == labels).sum().item()
            y_true.append(labels)
            y_pred.append(predicted)
        print('Accuracy of the network on the %d test sentences: %d %%' % (
            test_fetcher.num_samples, 100 * correct / total))

        y_true = torch.cat(y_true, 0)
        y_pred = torch.cat(y_pred, 0)
        print('F1 score: ', f1_score(y_true.numpy(), y_pred.numpy(), average='micro'))


print('===> Loading the trained model')
model = torch.load('./trained_model/supervised_transformer.pt')
print('Finished loading the trained model')

print('===> Start testing')
test(test_fetcher, model)
print('Finished testing')