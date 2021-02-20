#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[4]:


vocab_file='./data/glove.6B.200d.txt'
with open(vocab_file, encoding='utf-8') as f:
    print(len(f.readlines()))


# In[5]:


def get_embed(file):
    word2idx = {}
    row = 1
    word2embed = {}
    
    word2idx['PAD&UNK']=0
    word2embed['PAD&UNK']=[float(0)]*200
    
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
            
    idx2word = {idx:w for w, idx in word2idx.items()}
    idx2embed = {}
    for idx in range(len(word2idx)):
        idx2embed[idx] = word2embed[idx2word[idx]]
    #以 单词the 、索引 1、embed=[...]为例
    # word2idx['the'] = 1   idx2word[1] = 'the'
    # word2embed['the'] = [...]
        
    
    embed = np.array([idx2embed[idx] for idx in range(len(word2idx))])
    # embed = [ [...] , [...], ... , [...] ]
    return embed, word2idx, idx2word


# In[ ]:


embed, word2idx, idx2word=get_embed(vocab_file)
with open('./data/embed.pkl','wb') as f_embed:
    pickle.dump(embed,f_embed)
with open('./data/word2idx.pkl','wb') as f_w2i:
    pickle.dump(word2idx,f_w2i)
with open('./data/idx2word.pkl','wb') as f_i2w:
    pickle.dump(idx2word,f_i2w)


# In[6]:


with open('./data/word2idx.pkl','rb') as f_w2i:
    word2idx=pickle.load(f_w2i)
with open('./data/idx2word.pkl','rb') as f_i2w:
    idx2word=pickle.load(f_i2w)
with open('./data/embed.pkl','rb') as f_embed:
    embed=pickle.load(f_embed)
    
print(len(word2idx))


# In[7]:


text='the of to .'
print("将'the of to .'文本转化为id序列：",end="")
for word in text.lower().split():
    print(word2idx[word],end=' ')


# In[8]:


id_list=[19,15,4526,5873]
print("id_list转化为文本为：",end="")
for idx in id_list:
    print(idx2word[idx],end=" ")


# In[9]:


print("将'the of to .'文本转化为嵌入向量为：")
for word in text.lower().split():
    print(embed[word2idx[word]])
print("将id_list转化为嵌入向量为：")
for idx in id_list:
    print(embed[idx])


# In[10]:


def padding(text,maxlen):
    pad_text=[]
    for sentence in text:
        #建立一个形状符合输出的全0列表
        pad_sentence=np.zeros(maxlen).astype('int64')
        cnt=0
        for index in sentence:
            pad_sentence[cnt]=index
            cnt+=1
            if cnt== maxlen:
                break
        pad_text.append(pad_sentence.tolist())
    return pad_text


# In[11]:


text=[[5,4,15,12,7,7],
     list(range(130))]
pad_text=padding(text,30)
print(pad_text)


# In[12]:


def char_index(text_a,text_b,len_a,len_b):
    with open('./data/word2idx.pkl','rb') as f_w2i:
        word2idx=pickle.load(f_w2i)
    with open('./data/idx2word.pkl','rb') as f_i2w:
        idx2word=pickle.load(f_i2w)
    with open('./data/embed.pkl','rb') as f_embed:
        embed=pickle.load(f_embed)
    a_list,b_list=[],[]
    
    #对文件中的每一行
    for a_sentence,b_sentence in zip(text_a,text_b):
        a,b=[],[]
        
        #对每一行中的每一个单词
        for word in str(a_sentence).lower().split():
            if word in word2idx.keys():
                a.append(word2idx[word])
            else:
                a.append(0)  #没被收录的字被记作“UNK”
                
        for word in str(b_sentence).lower().split():
            if word in word2idx.keys():
                b.append(word2idx[word])
            else:
                b.append(0)
                
        a_list.append(a)
        b_list.append(b)
        
    a_list=padding(a_list,len_a)
    b_list=padding(b_list,len_b)
        
    return a_list,b_list


# In[13]:


ta=['I like machine learning','salkdj alskdj alkj lllsj .']
tb=["good good study , day day up",'asl laksl alsl als kalsk']
a_list,b_list=char_index(ta,tb,10,5)

print(a_list)
print(b_list)


# In[14]:


def load_char_data(filename,len_a,len_b):
    df=pd.read_csv(filename,encoding='utf-8',header=None ,sep='\t')
    text_a=df[0].values
    text_b=df[1].values
    label=df[2].values
    a_index,b_index = char_index(text_a,text_b,len_a,len_b)
    return np.array(a_index),np.array(b_index),np.array(label)


# In[15]:


a_index,b_index,label=load_char_data('./data/SNLI/snli-test.txt',20,20)


# In[16]:


len(a_index)


# In[ ]:




