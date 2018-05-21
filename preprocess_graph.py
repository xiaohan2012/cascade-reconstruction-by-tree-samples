import argparse
from graph_helpers import load_graph_by_name, get_edge_weights


def normalize_globally(g):
    print('global normlization')
    weights = get_edge_weights(g)
    deg = g.degree_property_map("out", weights)
    w_max = deg.a.max()
    new_g = g.copy()
    new_weights = get_edge_weights(new_g)
    new_weights.a /= w_max
    new_deg = new_g.degree_property_map("out", new_weights)

    # add self-loops
    self_loops = [(v, v) for v in new_g.vertices()]
    new_g.add_edge_list(self_loops)

    # assign new weights
    new_weights = get_edge_weights(new_g)
    for v, v in self_loops:
        new_weights[new_g.edge(v, v)] = 1 - new_deg[v]

    new_g.edge_properties['weights'] = new_weights
    return new_g


def reverse_edge_weights(g):
    print('reversing')
    weights = get_edge_weights(g)
    for e in g.edges():
        u, v = int(e.source()), int(e.target())
        if u < v:
            er = g.edge(e.target(), e.source())
            # print('before', weights[e], weights[er])
            weights[e], weights[er] = weights[er], weights[e]
            # print('after', weights[e], weights[er])
    g.edge_properties['weights'] = weights
    return g


def preprocess(g):
    return reverse_edge_weights(
        normalize_globally(g))


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-g', '--graph', help='graph name')
    parser.add_argument('-s', '--graph_suffix', default='', help='')
    parser.add_argument('-w', '--weighted', action='store_true', help='')
    parser.add_argument('-r', '--only_reserve', action='store_true', help='')
    parser.add_argument('-n', '--only_normalize', action='store_true', help='')
    parser.add_argument('-o', '--output_path', help='')
    
    args = parser.parse_args()
    
    g = load_graph_by_name(args.graph,
                           weighted=args.weighted,
                           suffix=args.graph_suffix)
    if args.only_reserve:
        print('only_reserve')
        new_g = reverse_edge_weights(g)
    elif args.only_normalize:
        print('only normlize')
        new_g = normalize_globally(g)
    else:
        new_g = preprocess(g)

    new_g.save(args.output_path)
    print('saved to {}'.format(args.output_path))
    

if __name__ == '__main__':
    main()
