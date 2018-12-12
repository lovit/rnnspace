# Recurrent Neural Network (RNN) based Space Correction

Comparison with RNN, LSTM, GRU, BiLSTM, BiLSTM-CRF

## Usage

For training, first scan characters and create character - index encoder

```python
from rnnspace import scan_vocabulary

texts = ['list of str type', 'sentence example']
idx_to_char, char_to_idx = scan_vocabulary(texts, min_count=1)
```

To prepare trainable data, encode character sequence to index sequence using sent_to_xy function.

```python
from rnnspace import sent_to_xy

X = [] # list of sentence
Y = [] # list of label

for text in texts:
    x, y = sent_to_xy(text, _vocab_to_idx)
    X.append(x)
    Y.append(y)
```

To train model,

```python
from rnnspace.models import LSTMSpace

# set parameters
embedding_dim = 16
hidden_dim = 64
vocab_size = len(idx_to_vocab) + 1 # for unknown character
tagset_size = 2
num_threads = 3

# model
model = LSTMSpace(embedding_dim, hidden_dim, vocab_size, tagset_size)
# loss function
loss_function = nn.NLLLoss()
# optimization
optimizer = optim.SGD(model.parameters(), lr=0.1)

# set max num of threads
torch.set_num_threads(num_threads)

# train
model = train(model, loss_function, optimizer, X, Y, epochs=50, use_gpu=False)
```

You can save trained model with pickle

```python
import pickle

path = 'modelpath'
with open(path, 'wb') as f:
    pickle.dump(model, f)
```

For correction

```python
from rnnspace import correct

sent = '이건진짜좋은영화 라라랜드진짜좋은영화'
print(correct(sent, char_to_idx, model))
```

```
'이건 진짜 좋은 영화 라라랜드 진짜 좋은영화'
```

## Memo

[access gate value](https://discuss.pytorch.org/t/access-gates-of-lstm-gru/12399/4)