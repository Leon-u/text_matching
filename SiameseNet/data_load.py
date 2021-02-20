#!/usr/bin/env python
# coding: utf-8

# # 数据载入模块
# 目标：给定数据集所在路径，返回整个文件经过处理后而成的id索引及标签

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


vocab_file = './data/token_vec_300.bin'


# ### 读取vocab_file文件，生成嵌入矩阵以及建立字符到索引，索引到字符之间的双向关系

# In[6]:


def get_embed(file):
    word2idx = {} # 词 -> id
    row = 1
    word2embed = {} # 词 -> 嵌入
    
    word2idx['PAD&UNK'] = 0
    word2embed['PAD&UNK'] = [float(0)]*300
    
    with open(file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            word2embed[word] = embed
            word2idx[word] = row
            row+=1
            
    idx2word = {idx: w for w, idx in word2idx.items()}
    id2embed = {}
    for ix in range(len(word2idx)):
        id2embed[ix] = word2embed[idx2word[ix]]

    embed = np.array([id2embed[ix] for ix in range(len(word2idx))])
    return embed, word2idx, idx2word


# In[9]:


embed, char2idx, idx2char = get_embed(vocab_file)
text = '我爱学习'
text_ids = []
id_list = [14, 34, 55, 12, 7, 2, 44, 58, 59]
print("将'我爱学习'文本转化为id序列",end="")
for char in text.lower():
    print(char2idx[char],end=' ')
    text_ids.append(char2idx[char])
print("\nid_list转化为嵌入向量为")
for idx in id_list:
    print(embed[idx],end="")


# In[13]:


def padding(text, maxlen=20):
    pad_text = []
    for sentence in text:
        pad_sentence = np.zeros(maxlen).astype('int64')
        cnt=0
        for index in sentence:
            pad_sentence[cnt]=index
            cnt+=1
            if cnt == maxlen:
                break
        pad_text.append(pad_sentence.tolist())
    return pad_text


# In[14]:


text=[[5,4,15,12,7,7],
     list(range(130))]
pad_text = padding(text)
print(pad_text)


# # 整合上两函数，给定批量的文本对，返回他们的索引id
# 考虑未出现过的单词与截断

# In[17]:


def char_index(text_a, text_b, file):
    embed, char2idx, idx2char = get_embed(vocab_file)
    a_list, b_list = [], []
    
    # 对文件中的每一行
    for a_sentence, b_sentence in zip(text_a, text_b):
        a, b = [], []
        
        # 对每一行中的每一个字
        for char in str(a_sentence).lower():
            if char in char2idx.keys():
                a.append(char2idx[char])
            else:
                a.append(0)
                
                
        for char in str(b_sentence).lower():
            if char in char2idx.keys():
                b.append(char2idx[char])
            else:
                b.append(0)
                
        a_list.append(a)
        b_list.append(b)
        
    a_list = padding(a_list)
    b_list = padding(b_list)
    
    return a_list, b_list


# ### 函数应用举例

# In[18]:


ta = ['我爱你','岇鎩']
tb=["再来一遍",'溌郶']
a_list,b_list=char_index(ta,tb,vocab_file)

print(a_list)
print(b_list)


# In[19]:


def load_char_data(filename, file):
    df = pd.read_csv(filename,encoding='utf-8', sep='\t')
    text_a = df['text_a'].values
    text_b = df['text_b'].values
    label = df['label'].values
    a_index, b_index = char_index(text_a, text_b, file)
    return np.array(a_index), np.array(b_index), np.array(label)


# In[20]:


a_index, b_index, label = load_char_data('./data/lcqmc_train.tsv',vocab_file)


# In[21]:


len(a_index)


# In[22]:


print(a_index[:17])


# In[23]:


b_index[:17]


# In[24]:


label[:17]


# In[ ]:




