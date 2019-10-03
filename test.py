import torch
import my_dataloader
import torch.utils.data as data
from sklearn.metrics import f1_score


def test_cnn():
    # Load the trained model
    print('===> Loading the model')
    net = torch.load('./trained_model/supervised_cnn_debug.pt')
    print('Finished loading the model')

    # load the test data set
    print('===> Loading test data set')
    testObj = my_dataloader.myDataSet('TEST_FILE_FULL.TXT')
    testloader = data.DataLoader(testObj, batch_size=4, shuffle=False, num_workers=4)
    print('Finished loading the test data set')

    # TODO: test the performance for each epoch(change to F1 score)
    # test the model on the test data set
    print('===> Start testing the model')
    correct = 0
    total = 0

    # matrix used for computing f1 score
    y_true = None
    y_pred = None

    with torch.no_grad():
        index = 0
        for d in testloader:
            inputs, labels = d
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)  # predicted shape: [batch_size, 1]
            total += labels.size(0)  # labels shape: [batch_size, 1]
            correct += (predicted == labels).sum().item()
            batch = predicted.shape[0]
            if index == 0:
                y_true = labels
                y_pred = predicted
            else:
                y_true = torch.cat((y_true, labels), 0)
                y_pred = torch.cat((y_pred, predicted), 0)
            index += 1
            # print(f1_score(labels.numpy(), predicted.numpy(), average='micro'))
            # <=> correct += torch.eq(predicted, labels).sum().item()

    print('Accuracy of the network on the %d test sentences: %d %%' % (
        testObj.num_samples, 100 * correct / total))

    print('F1 score: ', f1_score(y_true.numpy(), y_pred.numpy(), average='micro'))


def test_rnn():
    import data_loader
    # Load the trained model
    print('===> Loading the model')
    net = torch.load('./trained_model/supervised_rnn_v1.pt')
    print('Finished loading the model')

    # load the test data set
    print('===> Loading test data set')
    embed_dim = 50
    w2v_path = './data/word2vec.txt'
    sen_path = 'TEST_FILE_FULL.TXT'
    # sen_path = 'TRAIN_FILE.TXT'
    # sen_path = './data/nyt/test.txt'
    # sen_path = './data/nyt/train.txt'
    rel_path = './data/nyt/relation2id.txt'
    fetcher = data_loader.SemEvalFetcher(w2v_path, embed_dim, sen_path, is_shuffle=False, batch_size=4)
    # fetcher = data_loader.NYTFetcher(w2v_path, embed_dim, sen_path, rel_path)
    print('Finished loading the test data set')

    # TODO: test the performance for each epoch(change to F1 score)
    # test the model on the test data set
    print('===> Start testing the model')
    correct = 0
    total = 0

    # matrix used for computing f1 score
    y_true = None
    y_pred = None

    with torch.no_grad():
        index = 0
        for d in fetcher:
            inputs, labels = d
            labels = torch.from_numpy(labels)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)  # predicted shape: [batch_size, 1]
            total += labels.size(0)  # labels shape: [batch_size, 1]
            # print('predicted: ', predicted, ' labels: ', labels)
            correct += (predicted == labels).sum().item()
            batch = predicted.shape[0]
            if index == 0:
                y_true = labels
                y_pred = predicted
            else:
                y_true = torch.cat((y_true, labels), 0)
                y_pred = torch.cat((y_pred, predicted), 0)
            index += 1
            # print(f1_score(labels.numpy(), predicted.numpy(), average='micro'))
            # <=> correct += torch.eq(predicted, labels).sum().item()

    print('Accuracy of the network on the %d test sentences: %d %%' % (
        fetcher.num_samples, 100 * correct / total))

    print('F1 score: ', f1_score(y_true.numpy(), y_pred.numpy(), average='micro'))


# test_rnn()
test_cnn()
