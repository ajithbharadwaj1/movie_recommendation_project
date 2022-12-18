#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies.head()


# # Important Columns

# In[7]:


#genres
#id
#keywords
#title
#overview
#cast
#crew


# In[8]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.head()


# In[10]:


movies.isna().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.loc[0,'genres']


# # Since it is a string of list we can't iterate hence using ast

# In[14]:


import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L


# In[15]:


movies["genres"]=movies["genres"].apply(convert)


# In[16]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[17]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[18]:


movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])
movies.head()


# In[19]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[20]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[21]:


movies.head()


# In[22]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()


# In[23]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[24]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[25]:


movies


# In[26]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[27]:


movies.head()


# In[28]:


new_df=movies[["movie_id","title","tags"]]
new_df


# In[29]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df


# In[31]:


new_df['tags']=new_df['tags'].apply(lambda x : x.lower())


# In[37]:


import nltk


# In[38]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[41]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[43]:


new_df["tags"]=new_df["tags"].apply(stem)


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[45]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[46]:


vector.shape


# In[47]:


cv.get_feature_names()


# In[48]:


from sklearn.metrics.pairwise import cosine_similarity


# In[49]:


similarity = cosine_similarity(vector)


# In[50]:


similarity


# In[51]:


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
        


# In[53]:


import pickle
pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[55]:


recommend("Pirates of the Caribbean: At World's End")


# In[ ]:




