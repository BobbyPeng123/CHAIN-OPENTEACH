#! /bin/bash

sub_sensor_baseline=false
use_sensor_diffs=false
history=false
usage_str="Usage: $0 [-o obs_type] [-a aux_keys] [-s sub_sensor_baseline<FLAG>] [-u use_sensor_diffs<FLAG>] [-r history<FLAG>]"
while getopts ":o:a:s:u:r:" option; do
  case $option in
    o)
      obs_type="$OPTARG"
      ;;
    a)
      aux_keys="$OPTARG"
      ;;
    s)
      sub_sensor_baseline=true
      ;;
    u)
      use_sensor_diffs=true
      ;;
    r)
      history=true
      ;;
    :) usage 1 "-$OPTARG requires an argument" ;;
    *)
      echo "$usage_str"
      exit 1
      ;;
  esac
done

if [ -z "$obs_type" ]; then
    echo 'Missing -o flag' >&2
    echo $usage_str
    exit 1
fi
if [ -z "$aux_keys" ]; then
    echo 'Missing -a flag' >&2
    echo $usage_str
    exit 1
fi
project="${obs_type}_${aux_keys}_${history}_${sub_sensor_baseline}_${use_sensor_diffs}"
cmd_str="python train_bc.py -m suite.aux_keys=[$aux_keys] suite.subtract_sensor_baseline=$sub_sensor_baseline suite.use_sensor_diffs=$use_sensor_diffs"

cmd_str+=" history=$history \
        seed=34,35,36 \
        hydra/launcher=submitit_local \
        hydra.sweep.dir='/vast/\${oc.env:USER}/crossmodal_repr/exp/$project/\${now:%Y.%m.%d}' \
        hydra.sweep.subdir='\${seed}' \
        hydra.launcher.gpus_per_node=1 \
        hydra.launcher.tasks_per_node=1 \
        hydra.launcher.cpus_per_task=10 \
        hydra.launcher.timeout_min=600 \
        hydra.launcher.submitit_folder='/vast/\${oc.env:USER}/crossmodal_repr/exp/$project/\${now:%Y.%m.%d}/.slurm' \
        hydra.launcher.nodes=1"

# eval $cmd_str
echo $cmd_str
