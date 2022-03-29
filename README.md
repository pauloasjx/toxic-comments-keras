# toxic-comments-keras

Testing various architectures for the kaggle [Toxic Comment Classification Challenge][challenge-link].

```
├── __init__.py
├── archs
│   ├── gru_max_avg_pool.py
│   ├── gru_max_avg_pool_attention.py
│   └── lstm_max_avg_pool.py
├── download_dataset.ipynb
├── gru_max_avg_pool_glove.840B.300d.ipynb
├── lstm_max_avg_pool_glove.840B.300d.ipynb
├── train.py
├── utils.py
└── weights
    ├── bigru.best.zip
    ├── bigru_simple.best.zip
    ├── bilstm.best.zip
    ├── bilstm_simple.best.zip
    ├── gru.best.zip
    ├── gru_simple.best.zip
    ├── lstm.best.zip
    ├── lstm_simple.best.zip
    ├── rnn_simple.best.zip
    └── weights_base.best.zip
```

[challenge-link]: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge