#! /bin/zsh

graph="grqc"
n_cascades=100
n_observation_rounds=1
# n_cascades=8
# n_observation_rounds=1
cascade_model="si"
graph_suffix="_d_0.1"

obs_method="uniform"
obs_fraction=0.5

# works for IC
min_size=100
max_size=1000

# works for SI
cascade_fraction=0.05
dataset_id="${graph}-m${cascade_model}-s${cascade_fraction}-o${obs_fraction}-om${obs_method}"

# copy from existing cascades
output_dir="cascade/${dataset_id}"
from_cascade_dir="cascade/${graph}-m${cascade_model}-s${cascade_fraction}-o${obs_fraction}-ombfs-head"

print "ouput to ${output_dir}"

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
	# -c ${from_cascade_dir}
