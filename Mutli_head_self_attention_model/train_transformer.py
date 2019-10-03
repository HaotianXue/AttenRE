import transformer_model
import torch.nn as nn
import torch.optim as optim
import data_loader
import torch

batch_size = 32
embed_dim = 300
pos_dim = 10
num_layers = 2
hidden_dim = 128
num_epoch = 1
w2v_path = 'glove.6B.300d.txt'
sen_path = './data/nyt/train.txt'
rel_path = './data/nyt/relation2id.txt'
print('===> Fetching the data')
fetcher = data_loader.ddsNYTFetcher(w2v_path=w2v_path, emb_dim=embed_dim, data_path=sen_path, rel_path=rel_path,
                                    max_sen_len=100, padding=True)
print('Finished fetching the data')

# define the model
print('===> Start building the model')
model = transformer_model.dds_Transformer(d_w=embed_dim,
                                          d_e=pos_dim,
                                          num_classes=fetcher.num_rel,
                                          hidden_dim=2 * (embed_dim + 2 * pos_dim),
                                          word_emb_weight=torch.from_numpy(fetcher.word_embedding))
print('Finished building the model')

# define a loss function and optimizer
print('===> Starting building the loss function and optimizer')
criterion = nn.BCEWithLogitsLoss()
# parameters = filter(lambda p: p.requires_grad, model.parameters())  # remove parameters that don't require gradients
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=3e-4, weight_decay=1e-5)
print('Finished building the loss function and optimizer')

max_len = len(list(enumerate(fetcher)))  # number of training examples (here: 288754)

# training procedure
print('===> Start Training')
for epoch in range(num_epoch):
    fetcher.reset()
    optimizer.zero_grad()
    running_loss = 0.0
    for i, (x, y, seq_len) in enumerate(fetcher):
        outputs = model(x)
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
torch.save(model, './trained_model/dds_transformer_v1.pt')
print('Finished saving the model')
