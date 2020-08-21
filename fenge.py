# -*- coding: utf-8 -*-
import jieba
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from testf_result.precision import pre
import tqdm
import util.utils as utils


with open(r"D:\bert_1\中医古籍674本_fenge.txt", mode="w", encoding="utf-8") as f_in:

     with open(r"D:\bert_1\中医古籍674本.txt", mode="r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                #gt = " ".join(jieba.lcut(line))
                line=list(line)
                gt = " ".join(line)
                f_in.write(gt)
f_in.close()
f.close()



