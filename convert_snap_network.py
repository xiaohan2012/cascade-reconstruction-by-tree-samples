import networkx as nx
from tqdm import tqdm

# for dataset in tqdm(('p2p-gnutella08', 'arxiv-hep-th',
#                      'enron-email', 'dblp-collab')):
for dataset in tqdm(('auto-sys', )):
    with open('data/{}/graph.txt'.format(dataset)) as f:
        g = nx.Graph()
        for l in f:
            u, v = map(int, l.split()[:2])
            g.add_edge(u, v)

        ccs = nx.connected_components(g)
        lcc = max(ccs, key=len)
        lcc_g = g.subgraph(lcc)
        lcc_g = nx.convert_node_labels_to_integers(lcc_g)
        # nx.write_gpickle(lcc_g, 'data/{}/graph.gpkl'.format(dataset))
        nx.write_graphml(lcc_g, 'data/{}/graph.graphml'.format(dataset))
