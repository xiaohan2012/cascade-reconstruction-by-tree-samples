import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

from collections import OrderedDict, Iterable
from cycler import cycler

from inference import infection_probability
from graph_tool.draw import graph_draw
from graph_helpers import (extract_nodes,
                           observe_uninfected_node,
                           remove_filters)
from helpers import infected_nodes, cascade_source


def lattice_node_pos(g, shape):
    pos = g.new_vertex_property('vector<float>')
    for v in g.vertices():
        r, c = int(int(v) / shape[1]), int(v) % shape[1]
        pos[v] = np.array([r, c])
    return pos


OBS, QUERY, DEFAULT = range(3)

SIZE_ZERO = 0
SIZE_SMALL = 10
SIZE_MEDIUM = 20
SIZE_LARGE = 30

COLOR_BLUE = (31/255, 120/255, 180/255, 1.0)
COLOR_RED = (1.0, 0, 0, 1.0)
COLOR_YELLOW = (255/255, 217/255, 47/255, 1.0)
COLOR_WHITE = (255/255, 255/255, 255/255, 1.0)
COLOR_ORANGE = (252/255, 120/255, 88/255, 1.0)
COLOR_PINK = (1.0, 20/255, 147/255, 1.0)
COLOR_GREEN = (50/255, 205/255, 50/255, 1.0)
COLOR_GREY = (0.5, 0.5, 0.5, 1.0)
COLOR_BLACK = (0, 0, 0, 1.0)

SHAPE_CIRCLE = 'circle'
SHAPE_PENTAGON = 'pentagon'
SHAPE_HEXAGON = 'hexagon'
SHAPE_SQUARE = 'square'
SHAPE_TRIANGLE = 'triangle'
SHAPE_PENTAGON = 'pentagon'


def visualize(g, pos,
              node_color_info={},
              node_shape_info={},
              node_size_info={},
              edge_color_info={},
              edge_pen_width_info={},
              node_text_info={},
              color_map=mpl.cm.Reds,
              ax=None,
              output=None):

    def populate_property(dtype, info, on_edge=False):
        if on_edge:
            prop = g.new_edge_property(dtype)
        else:
            prop = g.new_vertex_property(dtype)
            
        prop.set_value(info['default'])
        del info['default']
        
        for entries, v in info.items():
            if on_edge:
                for n in entries:
                    prop[g.edge(*n)] = v
            else:
                if dtype not in {'int', 'float'}:
                    for n in entries:
                        prop[n] = v
                else:
                    prop.a[list(entries)] = v

        return prop

    # vertex color is a bit special
    # can pass both ndarray and RGB
    # for ndarray, it converted to cm.Reds

    # vertex_fill_color.set_value(node_color_info['default'])
    if isinstance(node_color_info, dict) and 'default' in node_color_info:
        del node_color_info['default']

    # colormap to convert to rgb
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    m = cm.ScalarMappable(norm=norm, cmap=color_map)

    if not isinstance(node_color_info, np.ndarray):
        assert isinstance(node_color_info, dict)
        vertex_fill_color = g.new_vertex_property('vector<float>')
        for entries, v in node_color_info.items():
            if isinstance(v, np.ndarray):
                assert len(entries) == len(v)
                for e, vv in zip(entries, v):
                    # convert to RGB
                    vertex_fill_color[e] = m.to_rgba(vv)
            else:
                for e in entries:
                    vertex_fill_color[e] = v
    else:
        vertex_fill_color = g.new_vertex_property('float')
        vertex_fill_color.a = node_color_info

    vertex_size = populate_property('int', node_size_info)
    vertex_shape = populate_property('string', node_shape_info)
    vertex_text = populate_property('string', node_text_info)
    
    edge_color = populate_property('string', edge_color_info, True)
    edge_pen_width = populate_property('float', edge_pen_width_info, True)
    
    graph_draw(g, pos=pos,
               vertex_fill_color=vertex_fill_color,
               vertex_size=vertex_size,
               vertex_shape=vertex_shape,
               edge_color=edge_color,
               edge_pen_width=edge_pen_width,
               vertex_text=vertex_text,
               mplfig=ax,
               vcmap=color_map,
               bg_color=[256, 256, 256, 256],
               output=output)


def default_plot_setting(g, c, X,
                         size_multiplier=1.0, edge_width_multiplier=1.0,
                         deemphasize_hidden_infs=False):
    source = cascade_source(c)
    inf_nodes = infected_nodes(c)
    hidden_infs = set(inf_nodes) - set(X)

    node_color_info = OrderedDict()
    node_color_info[tuple(X)] = COLOR_BLUE
    if not deemphasize_hidden_infs:
        node_color_info[tuple(hidden_infs)] = COLOR_YELLOW
    node_color_info[(source, )] = COLOR_GREEN
    node_color_info['default'] = COLOR_WHITE

    node_shape_info = OrderedDict()
    node_shape_info[tuple(X)] = SHAPE_SQUARE
    node_shape_info['default'] = SHAPE_CIRCLE
    node_shape_info[(source, )] = SHAPE_PENTAGON

    node_size_info = OrderedDict()

    node_size_info[tuple(X)] = 15 * size_multiplier
    node_size_info[(source, )] = 20 * size_multiplier
    if not deemphasize_hidden_infs:
        node_size_info[tuple(hidden_infs)] = 12.5 * size_multiplier
    node_size_info['default'] = 5 * size_multiplier

    node_text_info = {'default': ''}
    
    edge_color_info = {
        'default': 'white'
    }
    edge_pen_width_info = {
        'default': 2.0 * edge_width_multiplier
    }
    return {
        'node_color_info': node_color_info,
        'node_shape_info': node_shape_info,
        'node_size_info': node_size_info,
        'edge_color_info': edge_color_info,
        'edge_pen_width_info': edge_pen_width_info,
        'node_text_info': node_text_info
    }


def tree_plot_setting(g, c, X, tree_edges, color='red', **kwargs):
    s = default_plot_setting(g, c, X, **kwargs)
    s['edge_color_info'][tree_edges] = color
    return s


def query_plot_setting(g, c, X, qs,
                       node_size=20,
                       node_shape=SHAPE_TRIANGLE,
                       indicator_type='text',
                       color=1.0, **kwargs):
    s = default_plot_setting(g, c, X, **kwargs)
    s['node_shape_info'][tuple(qs)] = node_shape
    s['node_size_info'][tuple(qs)] = node_size
    # s['node_color_info'][tuple(qs)] = color

    if isinstance(indicator_type, str):
        indicator_types = {indicator_type}
    elif isinstance(indicator_type, Iterable):
        indicator_types = set(indicator_type)
    else:
        indicator_types = []
        
    if 'text' in indicator_types:
        for i, q in enumerate(qs):
            s['node_text_info'][tuple([q])] = str(i)
        
    if 'color' in indicator_types:
        color_depth = np.zeros(len(qs), dtype=np.float)

        for i, q in enumerate(qs):
            color_depth[i] = i / len(qs)

        s['node_color_info'][tuple(qs)] = color_depth

    return s


def heatmap_plot_setting(g, c, X, weight, **kwargs):
    inf_nodes = infected_nodes(c)
    hidden_infs = set(inf_nodes) - set(X)

    multipler = kwargs.get('size_multiplier', 1.0)
    s = default_plot_setting(g, c, X, **kwargs)
    if False:
        s['node_size_info'][tuple(X)] = 15
        s['node_size_info'][tuple(hidden_infs)] = 15
        s['node_size_info']['default'] = 7.5
    else:
        s['node_size_info'][tuple(X)] = 10 * multipler
        s['node_size_info'][tuple(hidden_infs)] = 10 * multipler
        s['node_size_info']['default'] = 10 * multipler
    
    s['node_color_info'] = weight
    return s


class QueryIllustrator():
    """illustrate the querying process"""

    def __init__(self, g, obs, c,
                 pos,
                 output_size=(300, 300),
                 vertex_size=20,
                 vcmap=mpl.cm.Reds):
        self.g_bak = g
        self.g = remove_filters(g)  # refresh
        self.obs = obs
        self.c = c

        self.pos = pos
        self.output_size = output_size
        self.vertex_size = vertex_size
        self.vcmap = vcmap
        
        self.inf_nodes = set((self.c >= 0).nonzero()[0])
        
        self.obs_inf = set(self.obs)
        self.obs_uninf = set()
        self.hidden_inf = self.inf_nodes - self.obs_inf
        self.hidden_uninf = set(extract_nodes(g)) - self.inf_nodes
        
    def add_query(self, query):
        if self.c[query] >= 0:  # infected
            self.obs_inf |= {query}
            self.hidden_inf -= {query}
        else:
            self.obs_uninf |= {query}
            self.hidden_uninf -= {query}
            observe_uninfected_node(self.g, query, self.obs_inf)

    def plot_snapshot(self, query, n_samples, ax=None):
        """plot one snap shot using one query and update node infection/observailability
        n_samples: num of samples used for inference
        """
        self.add_query(query)
        probas = infection_probability(self.g, self.obs_inf, n_samples=n_samples)
        # print(probas.shape)
        vcolor = self.node_colors(probas)
        vcolor[query] = 1  # highlight query

        vshape = self.node_shapes(query)
        vshape[query] = SHAPE_PENTAGON  # hack, override it

        graph_draw(self.g_bak,  # use the very earliest graph
                   pos=self.pos,
                   vcmap=self.vcmap,
                   output_size=self.output_size,
                   vertex_size=self.vertex_size,
                   vertex_fill_color=vcolor,
                   vertex_shape=vshape,
                   mplfig=ax)
        
    def node_properties_by_group(self, g, value_groups, dtype, default_val):
        """
        value_groups: dict of (dtype, list): (value, list of nodes)
        """
        vprop = g.new_vertex_property(dtype)
        vprop.set_value(default_val)
        for val, grp in value_groups:
            for i in grp:
                vprop[i] = val
        return vprop

    def node_shapes(self, query):
        groups = [(SHAPE_PENTAGON, [query]),
                  (SHAPE_SQUARE, self.obs_inf | self.obs_uninf),
                  (SHAPE_CIRCLE, self.hidden_inf),
                  (SHAPE_TRIANGLE, self.hidden_uninf)]
        return self.node_properties_by_group(self.g_bak, groups, 'string', SHAPE_CIRCLE)

    def node_colors(self, probas):
        color = self.g_bak.new_vertex_property('float')
        assert len(probas) == len(color.a)
        color.a = probas

        # might be fewer vertices in g than g_bak
        # because of vertex removal
        # for v, proba in zip(self.g.vertices(), probas):
        #     color[v] = proba
        return color


class InfectionProbabilityViz():
    def __init__(self, g,
                 pos,
                 output_size=(300, 300),
                 vcmap=mpl.cm.Reds):
        self.g = g
        self.pos = pos
        self.output_size = output_size
        self.vcmap = vcmap

    def plot(self, c, X, probas, interception_func=None, setting_kwargs={},
             **kwargs):
        setting = heatmap_plot_setting(self.g, c, X, probas,
                                       **setting_kwargs)
        if interception_func is not None:
            interception_func(setting)
        visualize(self.g, self.pos,
                  **setting,
                  **kwargs)
        

def set_cycler(ax):
    ax.set_prop_cycle(cycler('color', [COLOR_ORANGE, COLOR_PINK, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW,
                                       COLOR_BLACK, COLOR_GREY]) +
                      cycler('linestyle', ['-', ':', '--', '-.', '-', ':', '-']) +
                      cycler('marker', ['o', '*', '^', 'v', 's', 'd', 'p']) +
                      cycler('lw', [2, 2, 2, 2, 2, 2, 2]))



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """ 
     
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
        

    if window_len<3:
        return x
    
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y 
