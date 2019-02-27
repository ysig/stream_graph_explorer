import networkx as nx
from six import iteritems, itervalues
from utils import categorical
from collections import Counter, defaultdict
from itertools import combinations


# Robins measures 
## Nodestream

@categorical
def stats_node_stream_size(stream_graph):
    # Robin |W|: size of node stream
    return stream_graph.nodestream.shape[0]

@categorical
def stats_node_stream_coverage(stream_graph):
    # |W|/|V x T|: coverage of nostream
    return stream_graph.nodestream.shape[0]/\
        (stream_graph.nodeset.shape[0] * 
         stream_graph.timeset.shape[0])

@categorical
def time_number_of(streamgraph):
    # Robin W(u, .) : Time Duration of Node
    nodestream = streamgraph.nodestream
    data = Counter()
    for (u, _) in nodestream[['u', 'ts']].itertuples(index=False, name=None):
        data[u] += 1
    return data

@categorical
def time_coverage_of(streamgraph):
    # Robin W(u, .)/|T| : Time Coverage of Node
    # Assume that every element of a node-stream belongs in a timeset
    # Then node-density = #distinct-timestamps-per-node/#timeset-size
    timeset, nodestream = streamgraph.timeset, streamgraph.nodestream
    data = Counter()
    for (u, _) in nodestream[['u', 'ts']].itertuples(index=False, name=None):
        data[u] += 1
    size = timeset.shape[0]
    for u in data.keys():
        data[u] /= float(size)
    return data

@categorical
def node_number_at(streamgraph):
    # Robin W(., t) : Node Number at time instant t
    nodestream = streamgraph.nodestream
    data = Counter()
    for (u, ts) in nodestream[['u', 'ts']].itertuples(index=False, name=None):
        data[ts] += 1
    return data


@categorical
def node_density_at(streamgraph):
    # Robin W(., t)/|T| : Node Density at time instant t
    nodeset, nodestream = streamgraph.nodeset, streamgraph.nodestream
    data = Counter()
    for (_, ts) in nodestream[['u', 'ts']].itertuples(index=False, name=None):
        data[ts] += 1
    size = nodeset.shape[0]
    for ts in data.keys():
        data[ts] /= float(size)
    return data

## Linkstream
def Eprime(streamgraph):
    data = defaultdict(set)
    for (u, ts) in streamgraph.nodestream[['u', 'ts']].itertuples(index=False, name=None):
        data[ts].add(u)
    return {ts: st for ts, st in iteritems(data) if len(st) >= 2}

def Eprime_size(streamgraph):
    data = Counter()
    for (u, ts) in streamgraph.nodestream[['u', 'ts']].itertuples(index=False, name=None):
        data[ts] += 1 # each unit represent a distinct node at this time stamp
    # number of elements in a clique
    return sum(d**2/2.0 for d in itervalues(data) if d >= 2)

@categorical
def stats_link_stream_size(streamgraph):
    # Robin |E| size of link stream (in link-seconds / in link-instants)
    return streamgraph.linkstream.shape[0]

@categorical
def stats_link_stream_coverage(streamgraph):
    # Robin |E|/|E'| coverage of link stream (in %)
    return streamgraph.linkstream.shape[0]/Eprime_size(streamgraph)

@categorical
def neighbor_number_of(streamgraph, direction='out'):
    # Robin |E(u, ., .)| = neighbour duration / neighbour number of node
    # v ∈ V (in node-seconds / in node-instants)
    linkstream = streamgraph.linkstream
    data = Counter()
    if direction == 'out':
        def add(u, v):
            data[u] += 1
    elif direction == 'in':
        def add(u, v):
            data[v] += 1
    else:
        def add(u, v):
            data[u] += 1
            data[v] += 1

    for (u, v, _, _) in linkstream[['u', 'v', 'ts', 'w']].itertuples(index=False, name=None):
        add(u, v)
    return data

@categorical
def neighor_coverage_of(streamgraph, direction='out'):
    # Robin
    # |E(v, ., .)|/|E'(v, ., .)| neighbour coverage of node v ∈ V (in %)
    data = link_size_per_node(streamgraph, direction, False)
    data_ep = Counter()
    for ts, nodes in iteritems(Eprime(streamgraph)):
        for n in nodes:
            data_ep[n] += len(nodes) - 1
    return Counter(data[k]/float(data_ep[k]) for k in data_ep.keys())

@categorical
def link_number_at(streamgraph):
    # Robin
    # |E(., ., t)| link number at time instant t ∈ T (in links)
    linkstream = streamgraph.linkstream
    data = Counter()
    for (_, _, ts, _) in linkstream[['u', 'v', 'ts', 'w']].itertuples(index=False, name=None):
        data[ts] += 1
    return data

@categorical
def link_density_at(streamgraph):
    # Robin
    # |E(., ., t)|/|E'(., ., t)| link number at time instant t ∈ T (in links)
    data = link_number_at(streamgraph)
    data_result = Counter()
    for ts, nodes in iteritems(Eprime(streamgraph)):
        nom = data[ts]
        if nom > .0:
            data_result[ts] = nom / ((len(nodes)-1)**2/2.0)
    return data_result

@categorical
def link_number_of_node_couple(streamgraph, direction='out', with_weights=False):
    if direction == 'out':
        def hash(u, v):
            return (u, v)
    elif direction == 'in':
        def hash(u, v):
            return (v, u)
    else:
        def hash(u, v):
            return tuple(sorted([u, v]))
    linkstream, data = streamgraph.linkstream, Counter()
    for (u, v, _, w) in linkstream[['u', 'v', 'ts', 'w']].itertuples(index=False, name=None):
        if u != v:
            data[hash(u, v)] += w
    return data

@categorical
def link_coverage_of_node_couple(streamgraph, direction='out', with_weights=False):
    if direction == 'out':
        def hash(u, v):
            return (u, v)
    elif direction == 'in':
        def hash(u, v):
            return (v, u)
    else:
        def hash(u, v):
            return tuple(sorted([u, v]))
    def new():
        return [0, 0]
    data, Ep = defaultdict(new), Eprime(streamgraph)
    for (u, v, ts, w) in streamgraph.linkstream[['u', 'v', 'ts', 'w']].itertuples(index=False, name=None):
        if u != v:
            id = hash(u, v)
            data[id][0] += 1
            data[id][1] += int((u in Ep[ts]) and (v in Ep[ts]))
    return {k: v[0] / float(v[1]) for k, v in iteritems(data) if v[0] > .0 and v[1] > .0}

@categorical
def neighbor_number_of_at(streamgraph, direction='out'):
    # Robin |E(u, ., t)| = neighbour number of node u in V
    linkstream = streamgraph.linkstream

    data = Counter()
    if direction == 'out':
        def add(u, v, ts):
            data[ts][u] += 1
    elif direction == 'in':
        def add(u, v, ts):
            data[ts][v] += 1
    else:
        def add(u, v, ts):
            data[ts][u] += 1
            data[ts][v] += 1

    data = defaultdict(Counter)
    for (u, v, ts, _) in streamgraph.linkstream[['u', 'v', 'ts', 'w']].itertuples(index=False, name=None):
        add(u, v, ts)
    return data

@categorical
def neighbor_density_of_at(streamgraph, direction='out'):
    # Robin |E(u, ., t)| = neighbour number of node u in V
    linkstream = streamgraph.linkstream
    data, ep = neighbor_number_of_at(streamgraph, direction), Eprime(streamgraph)
    data_result = defaultdict(Counter)
    for t, data_u in iteritems(data_result):
        nodes = ep[t]
        for node, value in iteritems(data_u):
            if node in nodes:
                data_result[t][u] = value/float(len(nodes) - 1)
    return data_result

# Extras
@categorical
def link_load_per_node(streamgraph):
    # Count the total weight of each node
    linkstream = streamgraph.linkstream
    data = Counter()
    for (u, v, _, w) in linkstream[['u', 'v', 'ts', 'w']].itertuples(index=False, name=None):
        data[u] += w
    return data


# Stats
@categorical
def stats_number_of_nodes(stream_graph):
    return stream_graph.nodeset.shape[0]

@categorical
def stats_number_of_links(stream_graph):
    return len(set(stream_graph.linkstream[['u', 'v']].itertuples(index=False, name=None)))

@categorical
def stats_number_of_time_instants(stream_graph):
    return len(set(stream_graph.timeset[['ts']].itertuples(index=False, name=None)))

@categorical
def stats_number_of_interactions(stream_graph):
    return stream_graph.linkstream[['ts']].shape[0]
