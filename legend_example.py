
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt

from viz_helpers import COLOR_BLUE, COLOR_WHITE, COLOR_YELLOW, COLOR_ORANGE, COLOR_GREEN, COLOR_PINK


# In[26]:


def plot_cbar(labels, output, ticks=[0, 1]):
    plt.style.use('paper')
    a = np.array([[0,1]])
    fig = plt.figure(figsize=(9, 1.5))
    img = plt.imshow(a, cmap="Reds")
    plt.gca().set_visible(False)
    cax = plt.axes([0.11, 0.3, 0.8, 0.2])
    cbar = plt.colorbar(orientation="horizontal", cax=cax, ticks=ticks)
    cbar.ax.set_xticklabels(labels)
    cbar.outline.set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.tight_layout()
    plt.savefig(o)


# In[27]:


inputs = [(['queried early', 'queried late'], 'figs/inspect-query-process/cbar.pdf'),
           (['P(infected)=0', 'P(infected)=1'], 'figs/intro/cbar.pdf')]
for labels, o in inputs:
    plot_cbar(labels, o)


# In[23]:


plt.style.use('paper')

import matplotlib.pyplot as plt

def plot_query_legend(output):

    appearance_configs = [
        (COLOR_BLUE, 's', 20), (COLOR_YELLOW, 'o', 20), (COLOR_ORANGE, '^', 20), 
        (COLOR_GREEN, 'p', 20), (COLOR_WHITE, 'o', 10)
    ]
    labels = ['observed infected', 'hidden infected', 'query', 'source', 'hidden uninfected']

    handles = []
    for config in appearance_configs:
        color, shape, size = config
        h = plt.plot([],[],marker=shape, color=color, 
                     markersize=size,
                     ls="none")[0]
        handles.append(h)

    legend = plt.legend(handles, labels, loc=2,
                        framealpha=1, frameon=False,
                        ncol=3,
                        numpoints=1)
    plt.axis('off')
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(output, dpi="figure", bbox_inches=bbox)
# plt.show()


# In[24]:


outputs = ['figs/inspect-query-process/legend.pdf',
           'figs/intro/legend.pdf']
for o in outputs:
    plot_query_legend(o)


# In[32]:


plt.style.use('paper')
ax = plt.subplot()
method_configs = [
    (COLOR_ORANGE, 'o', '-'),
    (COLOR_PINK, '*', ':'),
    (COLOR_BLUE, '^', '--'),
    (COLOR_GREEN, 'v', '-.'),
    (COLOR_YELLOW, 's', '-')
]
labels = ['random', 'pagerank', 'entropy', 'cond-entropy', 'mutual-info']
size = 32
handles = []
for config in method_configs:
    color, shape, linestyle = config
    h = ax.plot([],[], marker=shape, color=color,                  
                 linestyle=linestyle,
                 markersize=size)[0]
    handles.append(h)

legend = ax.legend(handles, labels, loc=2,
                   framealpha=1, frameon=False,
                   fontsize=32,
                   ncol=5)
ax.axis('off')
fig  = legend.figure
fig.canvas.draw()
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('figs/method_legend.pdf', dpi="figure", bbox_inches=bbox)


# In[31]:


plt.style.use('paper')
ax = plt.subplot()
method_configs = [
    (COLOR_ORANGE, '^', ':'),
    (COLOR_BLUE, 'v', '--'), 
]
labels = ['SI', 'community']
size = 32
handles = []
for config in method_configs:
    color, shape, linestyle = config
    h = ax.plot([],[], marker=shape, color=color,                  
                 linestyle=linestyle,
                 markersize=size)[0]
    handles.append(h)

legend = ax.legend(handles, labels, loc=2,
                   framealpha=1, frameon=False,
                   fontsize=32,
                   ncol=5)
ax.axis('off')
fig  = legend.figure
fig.canvas.draw()
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('figs/cascade_legend.pdf', dpi="figure", bbox_inches=bbox)

