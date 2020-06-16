#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import json


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack


import matplotlib.pyplot as plt
np.random.seed(0)


# In[13]:


df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
df.info()


# In[14]:


df['hds']=df['headline']+df['short_description']


# In[30]:


import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[31]:


stemmer = PorterStemmer()


# In[32]:


def text_process(mess):
    nopunc=[char for char in mess if char not in string.punctuation]
    new1=''.join(nopunc)
    new2=[stemmer.stem(word) for word in new1]
    new3=''.join(new2)
    return[word for word in new3.split()if word.lower()not in stopwords.words('english') ]



# In[33]:


df['hds'].head()


# In[34]:


df['hds'].head(5).apply(text_process)


# In[35]:


title_tr, title_te, category_tr, category_te = train_test_split(df['hds'],df['category'])
title_tr, title_de, category_tr, category_de = train_test_split(title_tr,category_tr)


# In[36]:


print("Training    : ",len(title_tr))
print("Developement: ",len(title_de),)
print("Testing     : ",len(title_te))


# In[37]:


from wordcloud import WordCloud


# In[ ]:


text = " ".join(title_tr)
wordcloud = WordCloud().generate(text)
plt.figure()
plt.subplots(figsize=(20,12))
wordcloud = WordCloud(
    background_color="white",
    max_words=len(text),
    max_font_size=40,
    relative_scaling=.5).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

