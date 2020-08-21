# -*- coding: gbk -*-
import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

from bert_pytorch import BERT
from bert_1.pretrain import BERTTrainer

USE_CUDA = torch.cuda.is_available()

device=torch.device('cuda' if USE_CUDA else 'cpu')
# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值


BATCH_SIZE = 32
EMBEDDING_SIZE = 650
MAX_VOCAB_SIZE = 50000
TEXT = torchtext.data.Field()
train= torchtext.datasets.LanguageModelingDataset(path=r"D:\bert_1\一得集_fenge.txt", text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocabulary size: {}".format(len(TEXT.vocab)))

VOCAB_SIZE = len(TEXT.vocab)
train_iter= torchtext.data.BPTTIterator(
   train, batch_size=BATCH_SIZE, device=-1, bptt_len=32, repeat=False, shuffle=True)
it = iter(train_iter)
batch = next(it)
print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:,0].data]))
print("--------------")
print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,0].data]))




output_path=r"D:\bert_1\bert_fenci_model"
hidden=256
layers=8
attn_heads=8
seq_len=20
batch_size=64,
epochs=10
num_workers=1
with_cuda=False
log_freq=10
corpus_lines=None
lr=1e-3
adam_weight_decay=0.01
adam_beta1=0.9

adam_beta2=0.999
print("Building BERT model")
bert = BERT(len(TEXT.vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads)

print("Creating BERT Trainer")
trainer = BERTTrainer(bert, len(TEXT.vocab), train_dataloader=train_iter, test_dataloader=None,
                      lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                      with_cuda=with_cuda, log_freq=log_freq)

print("Training Start")
for epoch in range(epochs):
    print("------------------")
    trainer.train(epoch)
    trainer.save(epoch, output_path)