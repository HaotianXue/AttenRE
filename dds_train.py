import torch
import torch.nn as nn
import torch.optim as optim
import model
import data_fetcher

'''
This file is used to train dds model
'''


def train_cnn():
    batch_size = 32
    num_epoch = 3
    embed_dim = 50
    pos_dim = 5
    num_filter = 200
    window_size = 3
    w2v_path = './data/word2vec.txt'
    rel_path = './data/nyt/relation2id.txt'
    sen_path = './data/nyt/train.txt'
    print('===> Fetching the data')
    fetcher = data_fetcher.NYTFetcher(w2v_path, rel_path, embed_dim, sen_path,
                                      is_shuffle=True, max_sen_len=100, padding=True)
    print('Finished fetching the data')

    # define the model
    print('===> Start building the model')
    net = model.ddsNet(d_w=embed_dim,
                       d_e=pos_dim,
                       num_filter=num_filter,
                       num_classes=fetcher.num_rel,
                       window_size=window_size,
                       word_emb_weight=torch.from_numpy(fetcher.word_embedding))
    print('Finished building the model')

    # define a loss function and optimizer
    print('===> Starting building the loss function and optimizer')
    criterion = nn.BCEWithLogitsLoss()
    # parameters = filter(lambda p: p.requires_grad, net.parameters())  # remove parameters that don't require gradients
    parameters = net.parameters()
    optimizer = optim.Adam(parameters, lr=3e-4, weight_decay=1e-5)
    print('Finished building the loss function and optimizer')

    max_len = len(list(enumerate(fetcher)))  # number of training examples (here: 288754)

    # training procedure
    print('===> Start Training')
    for epoch in range(num_epoch):
        fetcher.reset()
        optimizer.zero_grad()
        running_loss = 0.0
        for i, (x, y) in enumerate(fetcher):
            outputs = net(x)
            loss = criterion(outputs, torch.from_numpy(y).float()) / batch_size
            loss.backward()
            running_loss += loss.item()
            if i % batch_size == 0 or i == max_len - 1:
                optimizer.step()
                optimizer.zero_grad()
                # print('%d Done, loss = %f' % (i, loss))
                print('%d Done, loss = %f' % (i, running_loss))
                running_loss = 0.0

    print('Finished Training')
    # Save the model
    print('===> saving the model')
    torch.save(net, './trained_model/dds_cnn_v1.pt')
    print('Finished saving the model')


def train_rnn():
    batch_size = 32
    num_epoch = 5
    embed_dim = 50
    pos_dim = 5
    num_layers = 2
    hidden_dim = 128
    w2v_path = './data/word2vec.txt'
    rel_path = './data/nyt/relation2id.txt'
    sen_path = './data/nyt/train.txt'
    print('===> Fetching the data')
    fetcher = data_fetcher.NYTFetcher(w2v_path, rel_path, embed_dim, sen_path,
                                      is_shuffle=True, max_sen_len=100)
    print('Finished fetching the data')

    # define the model
    print('===> Start building the model')
    net = model.ddsRNN(d_w=embed_dim,
                       d_e=pos_dim,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       num_classes=fetcher.num_rel,
                       word_emb_weight=torch.from_numpy(fetcher.word_embedding))
    print('Finished building the model')

    # define a loss function and optimizer
    print('===> Starting building the loss function and optimizer')
    # TODO: try MultiLabelSoftMarginLoss
    criterion = nn.BCEWithLogitsLoss()
    # parameters = filter(lambda p: p.requires_grad, net.parameters())  # remove parameters that don't require gradients
    parameters = net.parameters()
    optimizer = optim.Adam(parameters, lr=3e-4, weight_decay=1e-5)
    print('Finished building the loss function and optimizer')

    max_len = len(list(enumerate(fetcher)))  # number of training examples (here: 288754)

    # training procedure
    print('===> Start Training')
    for epoch in range(num_epoch):
        fetcher.reset()
        optimizer.zero_grad()
        running_loss = 0.0
        for i, (x, y) in enumerate(fetcher):
            outputs = net(x)
            loss = criterion(outputs, torch.from_numpy(y).float()) / batch_size
            loss.backward()
            running_loss += loss.item()
            if i % batch_size == 0 or i == max_len - 1:
                optimizer.step()
                optimizer.zero_grad()
                # print('%d Done, loss = %f' % (i, loss))
                print('%d Done, loss = %f' % (i, running_loss))
                running_loss = 0.0

    print('Finished Training')
    # Save the model
    print('===> saving the model')
    torch.save(net, './trained_model/dds_rnn_v1.pt')
    print('Finished saving the model')


train_cnn()
