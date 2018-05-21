# coding: utf-8

import pandas as pd
from graph_tool import Graph
from tqdm import tqdm

EPS = 1e-13

graph = 'flixster'
sep = '\t'

df = pd.read_csv('data/{}/{}.txt'.format(graph, graph), sep=sep, header=None, names=['u', 'v', 'w'])


# In[24]:


df.head(5)


# In[25]:


g = Graph(directed=True)


# In[26]:


all_nodes = set(df['u'].as_matrix()) | set(df['v'].as_matrix())


# In[27]:


g.add_vertex(len(all_nodes))


# In[28]:


edges =  list(zip(df['u'].as_matrix(), df['v'].as_matrix()))


# In[29]:


g.add_edge_list(edges)


# In[30]:


# add 
edges_iter = list(g.edges())
for e in tqdm(edges_iter):
    u, v  = int(e.source()), int(e.target())
    if g.edge(v, u) is None:
        g.add_edge(v, u)


# In[31]:


weight = g.new_edge_property('float')
weight.set_value(EPS)


# In[32]:


for i, r in tqdm(df.iterrows(), total=df.shape[0]):
    u, v, w = int(r['u']), int(r['v']), r['w']
    weight[g.edge(u, v)] = w
g.edge_properties['weights'] = weight


# In[33]:


deg_out = g.degree_property_map('out', weight=weight)


# In[34]:


r = deg_out.a
r = r[r < 2]
s = pd.Series(r)
# s.hist(bins=30)


# In[35]:


g.save('data/{}/graph.gt'.format(graph))
g.save('data/{}/graph_weighted.gt'.format(graph))

