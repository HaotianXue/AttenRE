# Deep_Relation_Extraction
Extract relation within unstructured sentence by using deep learning techniques

A relation knowledge graph is consisted of directed graph where a node is a triple (entity_1, relation, entity_2).

Relation extraction task is to predict the relation between two entities within a sentence or a paragraph.

This project uses supervised learning and distant supervision learning to train deep relation extraction model.

Three models are implemented: CNN with position embedding, LSTM with attention and Multi-head self-attention.

Supervised learning: 
1) CNN: word embedding + position embedding + CNN + feed-forward neural network
2) RNN: word embedding + position embedding + LSTM + word-level attention + feed-forward neural network
3) Attention model: word embedding + position embedding + Multi-head self-attnetion encoder + word-level attention + feed-forward neural network

Distant supervision:
1) CNN: word embedding + position embedding + CNN + word-level attention + sentence-level attention + feed-forward neural network
2) RNN: word embedding + position embedding + LSTM + word-level attention + sentence-level attention + feed-forward neural network
3) Attention model: word embedding + position embedding + Multi-head self-attnetion encoder + word-level attention + sentence-level-attention + feed-forward neural network
