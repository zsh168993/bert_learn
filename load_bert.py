# -*- coding: utf-8 -*-
import torch
import tqdm
from torch.utils.data import DataLoader
from bert_1.dataset.vocab import WordVocab
from bert_1.dataset.dataset import BERTDataset
if __name__ == '__main__':

    #vocab = WordVocab(r"C:\Users\Administrator\Desktop\2.txt")
    #vocab.save_vocab(r"C:\Users\Administrator\Desktop\2.txt")
    vocab=WordVocab.load_vocab(r"D:\bert_1\伤寒杂病论.txt")

    data=BERTDataset( r"C:\Users\Administrator\Desktop\3.txt",vocab,seq_len=20, corpus_lines=None)
    train_data_loader = DataLoader(data, batch_size=64, num_workers=1)
    data_iter = tqdm.tqdm(
                          enumerate(train_data_loader),
                          total=len(train_data_loader),
                          bar_format="{l_bar}{r_bar}")
    checkpoint = torch.load(r"D:\bert_1\bert_out.txt.ep0")
    for i, data in data_iter:
        X= checkpoint.forward(data["bert_input"],data["segment_label"])
    print(X)
    #attention_matrices = model(text)
    #model.plot_attention(text, attention_matrices, layer_num=2, head_num=1)