import re
import os
import tqdm
#获取目标文件夹的路径
filedir = r"D:\中医古籍700本"
#获取当前文件夹中的文件名称列表
filenames=os.listdir(filedir)

filenames.sort()
f_in1 = open(r'D:\bert_1\中医古籍674本.txt', mode="w", encoding="utf-8")

#打开当前目录下的result.txt文件，如果没有则创建

   # 先遍历文件名
for filename in filenames:
    filepath = filedir+'/'+filename
    #遍历单个文件，读取行数
    print(filepath)
    with open(filepath,"r",encoding="GBK") as f:
        for line in tqdm.tqdm(f):
            f_in1.write(line,)
        f.close()

f_in1.close()
