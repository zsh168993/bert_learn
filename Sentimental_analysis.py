# -*- coding: utf-8 -*-
import torch
import tqdm
from torch.utils.data import DataLoader
from bert_1.dataset.vocab import WordVocab
from bert_1.dataset.sentiment_analysis import CLSDataset
import torch.nn  as nn


class Pretrain(nn.Module):
    def __init__(self,path=r"D:\bert_1\dataset\train_data.txt"):

        super(Pretrain, self).__init__()
        self.vocab = WordVocab.load_vocab(r"D:\bert_1\伤寒杂病论.txt")
        self.data = CLSDataset(path, self.vocab, max_seq_len=20, data_regularization=False)
        self.train_data_loader = DataLoader(self.data, batch_size=64, num_workers=0)

        self.Linear = nn.Linear(256, 1)
        self.Sigmoid = nn.Sigmoid()

        self.checkpoint = torch.load(r"D:\bert_1\bert_out.txt.ep0")
        self.parameters=self.Linear.parameters()
        self.optimizer=torch.optim.Adam([{'params': self.checkpoint.parameters()},{'params':self.Linear.parameters()},{'params':self.Sigmoid.parameters()}], lr=0.0001, weight_decay=1e-3)

    def compute_loss(self,predictions, labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        predictions = predictions.view(-1)
        labels = labels.float().view(-1)
        epsilon = 1e-8
        # 交叉熵
        loss = - labels * torch.log(predictions + epsilon) - (torch.tensor(1.0) - labels) * torch.log(
            torch.tensor(1.0) - predictions + epsilon)
        # 求均值, 并返回可以反传的loss
        # loss为一个实数

        loss = torch.mean(loss)
        return loss
    def train(self,train_flag=True):

         data_iter = tqdm.tqdm(
             enumerate(self.train_data_loader),
             # desc="EP_%s:%d" % (str_code, epoch),
             total=len(self.train_data_loader),
             bar_format="{l_bar}{r_bar}")
         for j,data in data_iter:

             sequence_output,X_ =self.checkpoint.forward(data["text_input"], data["label"])#X[64,20,256]
             sequence_output=sequence_output[2]
             sequence_output= sequence_output[:, 0]

             pooled =self.Linear(sequence_output)
             predictions=self.Sigmoid(pooled)
             loss=self.compute_loss(predictions,data["label"])


             if train_flag == True:
                 # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                 # 用获取的梯度更新模型参数
                self.optimizer.step()
             else:
                 for i in range(len(predictions)):
                     if predictions[i]>0.5:
                        print(self.train_data_loader.dataset.lines[i])
                        print(predictions[i])
                        print("正样本")
                        print("______________________________")

                     else:
                        print(self.train_data_loader.dataset.lines[i])
                        print(predictions[i])
                        print("负样本")
                        print("______________________________")
         print(loss.item())



if __name__ == '__main__':


    senti2 = Pretrain(r"D:\bert\corpus\hotel_feedbacks_sentiment\1.txt")
    senti2.load_state_dict(torch.load(r"D:\bert_1\Sentimental_analysis_model"))
    senti2.train(train_flag=False)