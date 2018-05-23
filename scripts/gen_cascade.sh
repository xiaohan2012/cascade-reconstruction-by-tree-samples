#! /bin/zsh

graph="grqc"
n_cascades=100
n_observation_rounds=1
# n_cascades=8
# n_observation_rounds=1
cascade_model="si"
graph_suffix="_0.1"

obs_method="uniform"
obs_fractions=(0.5 0.6 0.7 0.8 0.9)
obs_fractions=(0.5)
cascade_fractions=(0.05 0.1 0.15 0.2 0.25)
# cascade_fractions=(0.05)
# works for IC
min_size=100
max_size=1000

# works for SI

for obs_fraction in ${obs_fractions}; do
    for cascade_fraction in ${cascade_fractions}; do
	dataset_id="${graph}-m${cascade_model}-s${cascade_fraction}-o${obs_fraction}-om${obs_method}"

	# copy from existing cascades
	output_dir="cascade/${dataset_id}"
	from_cascade_dir="cascade/${graph}-m${cascade_model}-s${cascade_fraction}-o${obs_fraction}-ombfs-head"

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
		--min_size ${min_size} \
		--max_size ${max_size}
"
	    # -c ${from_cascade_dir}
	fi
    done
done
