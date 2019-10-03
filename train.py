import torch.nn as nn
import torch.optim as optim
import my_dataloader
import model
import torch
import torch.utils.data as data1
import data_loader
# for reproducibility
torch.manual_seed(1)

classes = ('Other', 'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
               'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
               'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
               'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
               'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
               'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
               'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
               'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
               'Content-Container(e1,e2)', 'Content-Container(e2,e1)'
               )


def train_cnn():

    # define hyper-parameter here
    d_w = 50
    d_e = 5
    num_filter = 200
    num_classes = 19
    window_size = 3

    # load in the dataset
    print('===> Start loading training data set')
    trainObj = my_dataloader.myDataSet('TRAIN_FILE.TXT')
    trainloader = data1.DataLoader(trainObj, batch_size=16, shuffle=True, num_workers=8)
    print('Finished loading the training data set')

    # define the model
    print('===> Start building the model')
    net = model.Net(d_w, d_e, num_filter, num_classes, window_size, trainObj.word_embedding)
    print('Finished building the model')

    # define a loss function and optimizer
    print('===> Starting building the loss function and optimizer')
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # parameters = filter(lambda p: p.requires_grad, net.parameters())  # remove parameters that don't require gradients
    parameters = net.parameters()  # pytorch 0.4.1 可以自动判别哪些parameters是require_grad=True
    optimizer = optim.Adam(parameters, lr=3e-4, weight_decay=1e-5)
    print('Finished building the loss function and optimizer')

    # training procedure
    print('===> Start Training')
    for epoch in range(13):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            '''
            print('pred: ', torch.max(outputs, 1)[1], outputs.type())
            print('label: ', labels, labels.type())
            '''
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Finished Training')

    # Save the model
    print('===> saving the model')
    torch.save(net, './trained_model/supervised_cnn_debug.pt')
    print('Finished saving the model')


def train_rnn():
    batch_size = 20
    num_epoch = 30
    embed_dim = 50
    pos_dim = 20
    num_layers = 1
    hidden_dim = 512
    w2v_path = './data/word2vec.txt'
    sen_path = 'TRAIN_FILE.TXT'
    # sen_path = './data/nyt/train.txt'
    rel_path = './data/nyt/relation2id.txt'
    print('===> Fetching the data')
    fetcher = data_loader.SemEvalFetcher(w2v_path, embed_dim, sen_path, is_shuffle=True, batch_size=batch_size)
    # fetcher = data_loader.NYTFetcher(w2v_path, embed_dim, sen_path, rel_path, is_shuffle=True, batch_size=batch_size)
    print('Finished fetching the data')

    # define the model
    print('===> Start building the model')
    net = model.RNN_net(d_w=embed_dim,
                        d_e=pos_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        num_classes=19,
                        word_emb_weight=torch.from_numpy(fetcher.word_embedding))
    print('Finished building the model')

    # define a loss function and optimizer
    print('===> Starting building the loss function and optimizer')
    criterion = nn.CrossEntropyLoss()
    # parameters = filter(lambda p: p.requires_grad, net.parameters())  # remove parameters that don't require gradients
    parameters = net.parameters()
    optimizer = optim.Adam(parameters, lr=0.01, weight_decay=1e-8)
    # optimizer = optim.Adadelta(parameters, lr=1e-4, weight_decay=1e-5)
    # optimizer = optim.Adagrad(parameters, lr=0.05)
    # optimizer = optim.SGD(parameters, lr=3e-4, momentum=0.9)
    print('Finished building the loss function and optimizer')

    max_len = fetcher.num_samples  # number of training examples (here: 8000)
    print('number of training examples: ', max_len)

    import test

    # training procedure
    print('===> Start Training')
    for epoch in range(num_epoch):
        fetcher.reset()
        optimizer.zero_grad()
        running_loss = 0.0
        for i, (x, y) in enumerate(fetcher):
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(x)
            y = torch.from_numpy(y)
            # print('pred: ', torch.max(outputs, 1)[1], outputs.type())
            # print('label: ', y, y.type())
            loss = criterion(outputs, y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(parameters, max_norm=2.7)
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            if i % 200 == 200 - 1:  # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                # print('===> saving the model')
                # torch.save(net, './trained_model/supervised_rnn_v1.pt')
                # print('Finished saving the model')
        # test.test_rnn()

    print('Finished Training')
    # Save the model
    print('===> saving the model')
    torch.save(net, './trained_model/supervised_rnn_v1.pt')
    print('Finished saving the model')


train_rnn()
#train_cnn()
