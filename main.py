# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from bert_1.model.bert import BERT
from bert_1.trainer.pretrain import BERTTrainer
from bert_1.dataset.dataset import BERTDataset
from bert_1.dataset.vocab import WordVocab

def train(train_dataset=r"D:\伤寒杂病论.txt",test_dataset=None,vocab_path=r"D:\bert_1\伤寒杂病论.txt",output_path=r"D:\bert_1\bert_out.txt.ep0",hidden=256,layers=8,attn_heads=8,seq_len=20,batch_size=64,
epochs=10,num_workers=1,with_cuda=False,log_freq=10,corpus_lines=None,lr=1e-3,adam_weight_decay=0.01,adam_beta1=0.9,adam_beta2=0.999):

    print("Loading Vocab", vocab_path)
    vocab = WordVocab.load_vocab(vocab_path)
    #vocab="Welcome to the \t the jungle\nI can stay \t here all night\n"
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", train_dataset)
    train_dataset = BERTDataset(train_dataset, vocab, seq_len=seq_len, corpus_lines=corpus_lines)

    print("Loading Test Dataset", test_dataset)
    test_dataset = BERTDataset(test_dataset, vocab,
                               seq_len=seq_len) if test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                          with_cuda=with_cuda, log_freq=log_freq)

    print("Training Start")
    for epoch in range(epochs):
       print("------------------")
       trainer.train(epoch)
       trainer.save(epoch, output_path)

    if test_data_loader is not None:
        trainer.test(epoch)
if __name__ == '__main__':
    train()
