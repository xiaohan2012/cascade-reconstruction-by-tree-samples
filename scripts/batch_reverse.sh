#! /bin/zsh

graphs=(grqc lattice-1024 infectious)

for graph in ${graphs}; do
    python3 preprocess_graph.py \
	    -g ${graph} \
	    -w \
	    -s "_uniform" \
	    -r \
	    -o data/${graph}/graph_weighted_uniform_rev.gt
done
