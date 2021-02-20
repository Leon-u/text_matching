#!/usr/bin/env python
# coding: utf-8

# # 词表生成模块
# 目标：给每个 n-gram 切片 一个独有的索引id

# In[2]:


def n_gram(word, n=3):
    s = []
    word = '#'+word+'#'
    for i in range(len(word)-(n-1)):
        s.append(word[i:i+3])
    return s

print(n_gram("keyboardkeyboard"))

def lst_gram(lst, n=3):
    s = []
    for word in str(lst).lower().split():
        s.extend(n_gram(word))
    return s

print(lst_gram("Two young children in blue jerseys"))


# # 读取文件
# 遍历文件中出现的每一个词，将其转化为n-gram切片后加入列表

# In[3]:


vocab = []


# In[4]:


file_path = "./MRPC/"
files = ["train_data.csv", "test_data.csv"]

for file in files:
    f = open(file_path+file, encoding='utf-8').readlines()
    for i in range(1, len(f)):
        s1, s2 = f[i][2:].strip('\n').split('\t')
        #对每一行，去掉前两个字符（标签（0，1，2）TAB键）
        #去掉末尾换行符，用TAB键切分字符串，刚好切为两个句子
        vocab.extend(lst_gram(s1))
        vocab.extend(lst_gram(s2))
        


# In[5]:


vocab = set(vocab)
print(vocab)


# In[6]:


vocab_list = ['[PAD]','[UNK]']
vocab_list.extend(list(vocab))


# In[7]:


list(vocab_list)


# In[8]:


print(len(vocab_list))


# In[9]:


vocab_file='vocab_mrpc.txt'
with open(vocab_file, 'w', encoding='utf-8') as f:
    for slice in vocab_list:
        f.write(slice)
        f.write('\n')


# In[ ]:




