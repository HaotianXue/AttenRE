import transformer_model
import torch.nn as nn
import torch.optim as optim
import data_loader
import torch

batch_size = 20
embed_dim = 300
pos_dim = 10
num_layers = 2
hidden_dim = 128
num_epoch = 1
w2v_path = 'glove.6B.300d.txt'
sen_path = './SemEval/TRAIN_FILE.txt'
print('===> Fetching the data')
fetcher = data_loader.SemEvalFetcher(w2v_path=w2v_path, emb_dim=embed_dim, data_path=sen_path,
                                     batch_size=batch_size, max_sen_len=100, padding=True)
print('Finished fetching the data')

'''
# define the model
print('===> Start building the model')
model = transformer_model.Transformer(d_w=embed_dim,
                                      d_e=pos_dim,
                                      num_classes=19,
                                      hidden_dim=2 * (embed_dim + 2 * pos_dim),
                                      word_emb_weight=torch.from_numpy(fetcher.word_embedding),
                                      num_layers=2,
                                      num_heads=4)
print('Finished building the model')
'''


print('===> Loading the trained model')
model = torch.load('./trained_model/supervised_transformer.pt')
print('Finished loading the trained model')


# define a loss function and optimizer
print('===> Starting building the loss function and optimizer')
criterion = nn.CrossEntropyLoss()
# parameters = filter(lambda p: p.requires_grad, model.parameters())  # remove parameters that don't require gradients
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=0.003, weight_decay=1e-5)
print('Finished building the loss function and optimizer')

# training procedure
print('===> Start Training')
for epoch in range(12):  # loop over the dataset multiple times
    fetcher.reset()
    running_loss = 0.0
    for i, data in enumerate(fetcher, 0):
        # get the inputs
        inputs, labels, seq_len = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, torch.from_numpy(labels))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 40 == 39:    # print every 40 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 40))
            running_loss = 0.0

print('Finished Training')

# Save the model
print('===> saving the model')
torch.save(model, './trained_model/supervised_transformer.pt')
print('Finished saving the model')
