#! /bin/zsh

# graphs=(lattice-1024)
graphs=(grqc fb-messages email-univ infectious lattice-1024)

n_cascades=100
n_observation_rounds=1
cascade_model="ic"
graph_suffix="_uniform"
# graph_suffix="_0.1"

obs_method="uniform"

obs_fractions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# obs_fractions=(0.5)

# cascade_fractions=(0.1 0.2 0.3 0.4 0.5)
cascade_fractions=(0.1)

for graph in ${graphs}; do
    for obs_fraction in ${obs_fractions}; do
	for cascade_fraction in ${cascade_fractions}; do
	    dataset_id="${graph}-m${cascade_model}-s${cascade_fraction}-o${obs_fraction}-om${obs_method}"

	    # copy from existing cascades
	    output_dir="cascade-with-edges/${dataset_id}"
	    from_cascade_dir="cascade-with-edges/${graph}-m${cascade_model}-s${cascade_fraction}-o${obs_fraction}-ombfs-head"

	    # print "ouput to ${output_dir}"
	    if [ ! -f "${output_dir}/99.pkl" ]; then
		print "
	python3 simulate_cascades.py \
		-g ${graph} \
		-n ${n_cascades} \
		-o ${obs_fraction} \
		-f ${graph_suffix} \
		--n_observation_rounds ${n_observation_rounds} \
		--use_edge_weights \
		-m ${cascade_model} \
		-d ${output_dir} \
		-s ${cascade_fraction} \
		--observation_method ${obs_method} \
                --store_tree
"
	    fi
	done
    done
done
