import plotly.graph_objs as go
import numpy as np
import networkx as nx
from collections import Counter
from six import iteritems

mean_color = 'rgba(88, 24, 69, 0.5)'
median_color = 'rgba(34, 64, 19, 0.5)'
min_color = 'rgba(88, 24, 69, 0.5)'
max_color = 'rgba(88, 24, 69, 0.5)'

# 1D

def barplot(data,
        x_map=None,
        name=None,
        sort_by_x=False,
        map_x_before_sorting=True,
        plot_min=False,
        plot_max=False,
        plot_mean=False,
        plot_median=False):
    xs, ys = list(), list()
    for x, y in iteritems(data):
        xs.append(x)
        ys.append(y)
    
    def map_x(xs):
        if x_map is not None:
            if isinstance(x_map, dict):
                xs = [x_map[x] for x in xs]
            elif isinstance(x_map, callable):
                xs = [x_map(x) for x in xs]
            else:
                raise ValueError('x_map should be dict or callable')
        return xs

    if map_x_before_sorting:
        xs = map_x(xs)

    xs, ys = np.array(xs), np.array(ys)
    if sort_by_x:
        idx = np.argsort(xs)[::-1]
    else:
        idx = np.argsort(ys)[::-1]
    xs, ys = xs[idx].tolist(), ys[idx].tolist()

    if not map_x_before_sorting:
        xs = map_x(xs)
    
    if x_map is None:
        ids, names = zip(*enumerate(xs))
    else:
        ids, names = zip(*((i, x) for (i, x) in enumerate(xs)))
    ids, names = list(ids), list(names)

    extra = []
    if plot_min:
        ysm = np.min(ys)
        line = dict(color=(min_color), width=1.5, dash='dash')
        extra.append(go.Scattergl(x=[ids[0], ids[-1]], y=[ysm, ysm], name=(("" if name is None else str(name) + '_') + 'min'), mode='lines', line=line))
    if plot_max:
        ysm = np.max(ys)
        line = dict(color=(max_color), width=1.5, dash='dash')
        extra.append(go.Scattergl(x=[ids[0], ids[-1]], y=[ysm, ysm], name=(("" if name is None else str(name) + '_') + 'max'), mode='lines', line=line))
    if plot_mean:
        ysm = np.mean(ys)
        line = dict(color=(mean_color), width=1.5, dash='dash')
        extra.append(go.Scattergl(x=[ids[0], ids[-1]], y=[ysm, ysm], name=(("" if name is None else str(name) + '_') + 'mean'), mode='lines', line=line))
    if plot_median:
        ysm = np.median(ys)
        line = dict(color=(median_color), width=1.5, dash='dash')
        extra.append(go.Scattergl(x=[ids[0], ids[-1]], y=[ysm, ysm], name=(("" if name is None else str(name) + '_') + 'median'), mode='lines', line=line))
    return [go.Bar(x=ids, y=ys, text=names, hoverinfo="y+text", showlegend=(name is not None), name=name)] + extra

def scatterplot_line(
        data,
        x_map=None,
        name=None,
        plot_min=False,
        plot_max=False,
        plot_mean=False,
        plot_median=False):
    kargs, xsl, ys = dict(), list(), list()
    for x, y in iteritems(data):
        xsl.append(x)
        ys.append(y)
    xs, ys = np.array(xsl), np.array(ys)
    kargs['hoverinfo'] = 'y'
    if x_map is not None:
        if isinstance(x_map, dict):
            kargs['text'] = [x_map[x] for x in xs]
        elif isinstance(x_map, callable):
            kargs['text'] = [x_map(x) for x in xs]
        else:
            raise ValueError('x_map should be dict or callable')
        kargs['hoverinfo'] += '+text'

    idx = np.argsort(xs)[::-1]
    xs, ys = xs[idx].tolist(), ys[idx].tolist()

    kargs['line'], extra = dict(shape='vh'), []
    if plot_min:
        ysm = np.min(ys)
        line = dict(color=(min_color), width=1.5, dash='dash')
        extra.append(go.Scattergl(x=[xs[0], xs[-1]], y=[ysm, ysm], name=(("" if name is None else str(name) + '_') + 'min'), mode='lines', line=line))
    if plot_max:
        ysm = np.max(ys)
        line = dict(color=(max_color), width=1.5, dash='dash')
        extra.append(go.Scattergl(x=[xs[0], xs[-1]], y=[ysm, ysm], name=(("" if name is None else str(name) + '_') + 'max'), mode='lines', line=line))
    if plot_mean:
        ysm = np.mean(ys)
        line = dict(color=(mean_color), width=1.5, dash='dash')
        extra.append(go.Scattergl(x=[xs[0], xs[-1]], y=[ysm, ysm], name=(("" if name is None else str(name) + '_') + 'mean'), mode='lines', line=line))
    if plot_median:
        ysm = np.median(ys)
        line = dict(color=(median_color), width=1.5, dash='dash')
        extra.append(go.Scattergl(x=[xs[0], xs[-1]], y=[ysm, ysm], name=(("" if name is None else str(name) + '_') + 'median'), mode='lines', line=line))
    return [go.Scattergl(x=xs, y=ys, mode='lines', name=name, showlegend=(name is not None), **kargs)] + extra

# 2M

def scatterplot_points(
        data,
        name=None,
        text_map=None,
        plot_min=False,
        plot_max=False,
        plot_mean=False,
        plot_median=False):
    data_x, data_y = data
    data_k = sorted(list(set(data_x.keys()) & set(data_y.keys())))
    x = [data_x[k] for k in data_k]
    y = [data_y[k] for k in data_k]
    
    kargs = dict()
    if text_map is not None:
        assert(isinstance(text_map, dict))
        data_k = [text_map[k] for k in data_k]
    kargs['text'] = data_k
    kargs['hoverinfo'] = 'x+y+text'

    extra = []
    if plot_min:
        marker = dict(color=(min_color))
        extra.append(go.Scattergl(x=np.min(x), y=np.min(y), name=(("" if name is None else str(name) + '_') + 'min'), mode='markers', marker=marker))
    if plot_max:
        marker = dict(color=(max_color))
        extra.append(go.Scattergl(x=np.max(x), y=np.max(y), name=(("" if name is None else str(name) + '_') + 'max'), mode='markers', marker=marker))
    if plot_mean:
        marker = dict(color=(mean_color))
        extra.append(go.Scattergl(x=np.mean(x), y=np.mean(y), name=(("" if name is None else str(name) + '_') + 'mean'), mode='markers', marker=marker))
    if plot_median:
        marker = dict(color=(median_color))
        extra.append(go.Scattergl(x=np.median(x), y=np.median(y), name=(("" if name is None else str(name) + '_') + 'median'), mode='markers', marker=marker))
    return [go.Scattergl(x=x, y=y, mode='markers', name=name, showlegend=(name is not None), **kargs)] + extra


# 2D

def heatmap(
        data,
        x_map=None,
        name=None):
    kargs = dict()
    xs = sorted(list(set((x for x, _ in data.keys()))))
    ys = sorted(list(set((y for _, y in data.keys()))))

    xs_map = {x: i for i, x in enumerate(xs)}
    ys_map = {y: i for i, y in enumerate(ys)}
    z = np.zeros(shape=(len(xs), len(ys)))
    for (x, y), v in iteritems(data):
        z[xs_map[x], ys_map[y]] = v
    if x_map is not None:
        if isinstance(x_map, dict):
            kargs['text'] = [[str((x_map[x], y)) for x in xs] for y in ys]
        elif isinstance(x_map, callable):
            kargs['text'] = [[str((x_map(x), y)) for x in xs] for y in ys]
        else:
            raise ValueError('x_map should be dict or callable')
    else:
        kargs['text'] = [[str((x, y)) for x in xs] for y in ys]
    kargs['hoverinfo'] = 'z+text'
    
    return [go.Heatmap(y=list(range(len(xs))), x=list(range(len(ys))), z=z.T, **kargs)]

def multi_scatterplot(
        data,
        x_map=None,
        name=None,
        plot_min=False,
        plot_max=False,
        plot_mean=False,
        plot_median=False):
    out = []
    ts = sorted(data.keys())
    vs = set(k for t, d in iteritems(data) for k in d.keys())
    vs_map = {v: i for i, v in enumerate(vs)}
    ts_map = {t: i for i, t in enumerate(ts)}
    zs = np.zeros(shape=(len(ts), len(vs)))
    for t, d in iteritems(data):
        for v, z in iteritems(d):
            zs[ts_map[t], vs_map[v]] = z

    kargs = {'hoverinfo': 'name'}
    if x_map is not None:
        if isinstance(x_map, dict):
            kargs['text'] = [x_map[t] for t in ts]
        elif isinstance(x_map, callable):
            kargs['text'] = [x_map(t) for t in ts]
        else:
            raise ValueError('x_map should be dict or callable')
        kargs['hoverinfo'] = 'text+' + kargs['hoverinfo']

    for names, i in iteritems(vs_map):
        out.append(go.Scattergl(x=ts, y=zs[:, i], mode="lines", line=dict(shape='vh'), showlegend=False, name=(names if name is None else str(names) + '_') + 'min', **kargs))

    if zs.size:
        if plot_min:
            line = dict(color=(max_color), width=1.5, dash='dash', shape='vh')
            out.append(go.Scattergl(x=ts, y=np.min(zs, axis=1), name=("min" if name is None else str(name) + '_' + 'min'), mode='lines', line=line, **kargs))
        if plot_max:
            line = dict(color=(max_color), width=1.5, dash='dash', shape='vh')
            out.append(go.Scattergl(x=ts, y=np.max(zs, axis=1), name=("max" if name is None else str(name) + '_' + 'max'), mode='lines', line=line, **kargs))
        if plot_mean:
            line = dict(color=(mean_color), width=1.5, dash='dash', shape='vh')
            out.append(go.Scattergl(x=ts, y=np.mean(zs, axis=1), name=("mean" if name is None else str(name) + '_' + 'mean'), mode='lines', line=line, **kargs))
        if plot_median:
            line = dict(color=(median_color), width=1.5, dash='dash', shape='vh')
            out.append(go.Scattergl(x=ts, y=np.median(zs, axis=1), name=("median" if name is None else str(name) + '_' + 'median'), mode='lines', line=line, **kargs))
    return out


def graphplot(data, node_colorscale='Rainbow', layout='spring', direction='out'):
    # colorscale options:
    #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
    #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
    #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
    if direction == 'out':
        graph = nx.DiGraph()
        graph.add_edges_from((u, v, {'weight': w}) for ((u, v), w) in iteritems(data))
    elif direction == 'in':
        graph = nx.DiGraph()
        graph.add_edges_from((v, u, {'weight': w}) for ((u, v), w) in iteritems(data))
    else:
        graph = nx.Graph()
        graph.add_edges_from((v, u, {'weight': w}) for ((u, v), w) in iteritems(data))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    print("#N =", graph.number_of_nodes(), "#E =", graph.number_of_edges())
    if layout == 'spring':
        layout = nx.spring_layout
    elif layout == 'spectral':
        layout = nx.spectral_layout
    elif layout == 'kamada-kawai':
        layout = nx.kamada_kawai_layout
    elif layout == 'random':
        layout = nx.random_layout
    elif layout == 'shell':
        layout = nx.shell_layout
    else:
        raise ValueError('unsupported layout: ' + layout)
    pos = layout(graph)
    print("Positions calculated")
    nodes = list(pos.keys())
    Xn, Yn = [pos[k][0] for k in nodes], [pos[k][1] for k in nodes]

    Xe, Ye, text_e = [], [], []
    max_d, degrees, wdegrees, weights = 0, Counter(), Counter(), []
    for u, v, w in graph.edges(data='weight'):
        Xe.append([pos[u][0], pos[v][0]])
        Ye.append([pos[u][1], pos[v][1]])
        weights.append(w)
        #degrees[u] += 1
        #wdegrees[u] += w
        if direction == 'both':
            #degrees[v] += 1
            #wdegrees[v] += w
            et = str(u) + "," + str(v) + ":" + str(w)
        else:
            et = str(u) + "->" + str(v) + ":" + str(w)
        text_e.append(et)

    #color = [degrees[k] for k in nodes]
    #opacity = [degrees[k]/float(max(color)) for k in nodes]
    #size = [wdegrees[k]+1 for k in nodes]
    opacity, size = 0.4, 5
    trace_nodes=dict(type='scatter',
                     x=Xn, 
                     y=Yn,
                     mode='markers',
                     text=nodes,
                     marker=dict(
                     #   showscale=True,
                     #   colorscale=node_colorscale,
                     #   reversescale=True,
                        opacity=opacity,
                     #   color=color,
                        size=size,
                     #colorbar=dict(
                     #       thickness=15,
                     #       title='Degree' + ("" if direction == 'both' else " (" + direction + ")"),
                     #       xanchor='left',
                     #       titleside='right'
                     #   ),
                     #   line=dict(width=2)),
                     ),
                     hoverinfo='text',
                     showlegend=False)

    #widths = [1.5*(w-min(weights))/float(max(max(weights) - min(weights), 1)) + 0.5 for w in weights]
    #opacities = [w/float(max(weights)) for w in weights]
    trace_edges= [dict(type='scatter', mode='lines',
                       x=x, y=y, name=te,
                       line=dict(width=1, opacity=1, color='#888'),
                       hoverinfo='name',
                       textposition='outside',
                       hoverlabel = dict(namelength = -1),
                       hoveron='fills',
                       showlegend=False) for x, y, te in zip(Xe, Ye, text_e)] 

    return trace_edges + [trace_nodes]
