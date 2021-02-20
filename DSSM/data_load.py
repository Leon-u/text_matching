#!/usr/bin/env python
# coding: utf-8

# # 数据载入模块
# 目标：给定数据集所在路径，返回整个文件经过处理后而成的id索引以及标签

# In[2]:


import numpy as np
import pandas as pd
import torch
from make_vocab import lst_gram, n_gram


# vocab_mrpc.txt文件，建立切片到索引，索引到切片间的双向关系

# In[4]:


def load_vocab():
    vocab = open('./vocab_mrpc.txt',encoding='utf-8').readlines()
    slice2idx = {}
    idx2slice = {}
    cnt = 0
    for char in vocab:
        char = char.strip('\n')
        slice2idx[char]=cnt
        idx2slice[cnt]=char
        cnt+=1
    return slice2idx, idx2slice


# ## 函数应用举例

# In[5]:


slice2idx, idx2slice=load_vocab()
text='do you like what you see'
text_ids=[]
id_list=[234,45,10,656,77,2091,2009,9872,254,555,10]

text_gram=lst_gram(text,n=3)
print("将'do you like what you see'文本转化为id序列：",end="")
for char in text_gram:
    print(slice2idx[char],end=' ')
    text_ids.append(slice2idx[char])
    
print("\nid_list转化为字符切片为：",end="")
for idx in id_list:
    print(idx2slice[idx],end=" ")


# ## 数据填充，截取，保证送入模型的数据格式规整

# In[7]:


def padding(text, maxlen=70):
    pad_text=[]
    for sentence in text:
        pad_sentence = np.zeros(maxlen).astype("int64")
        cnt=0
        for index in sentence:
            pad_sentence[cnt]=index
            cnt+=1
            if cnt==maxlen:
                break
        pad_text.append(pad_sentence.tolist())
    return pad_text


# In[8]:


# 函数应用举例
text=[[16,120,48,512],[234,45,10,656,77,2091],
     list(range(80))]
pad_text=padding(text)
print(pad_text)


# # 整合上两函数，给定批量的文本对，返回他们的索引id
# ## 考虑未出现过的单词与截断

# In[9]:


def char_index(text_a, text_b):
    slice2idx, idx2slice=load_vocab()
    a_list, b_list=[], []
    # 对文件中的每一行
    for a_sentence, b_sentence in zip(text_a, text_b):
        a, b = [], []
        # 对每一行中的每一个切片
        for slice in lst_gram(a_sentence):
            if slice in slice2idx.keys():
                a.append(slice2idx[slice])
            else:
                a.append(1)  #没被切片的记作UNK
                
        for slice in lst_gram(b_sentence):
            if slice in slice2idx.keys():
                b.append(slice2idx[slice])
            else:
                b.append(1)
                
        a_list.append(a)
        b_list.append(b)
        
    a_list=padding(a_list)
    b_list=padding(b_list)
    
    return a_list, b_list


# In[10]:


ta=['Families waiting in line at an amusement park','##)']
tb=['People are riding bikes.','#&^']
a_list,b_list=char_index(ta,tb)

print(a_list)
print(b_list)


# In[11]:


def load_char_data(filename):
    df = pd.read_csv(filename,sep='\t')
    text_a=df['#1 string'].values
    text_b=df['#2 string'].values
    label=df['quality'].values
    a_index, b_index = char_index(text_a, text_b)
    return np.array(a_index), np.array(b_index), np.array(label)


# In[12]:


a_index,b_index,label=load_char_data('./MRPC/test_data.csv')


# In[13]:


len(a_index)


# In[14]:


a_index[:7]


# In[15]:


label[0:7]

