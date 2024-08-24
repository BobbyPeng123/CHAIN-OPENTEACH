obs_type="pixels"
aux_keys="proprioceptive,sensor"
start_time=`echo date +%m%d-%H%M%S`
log_name="logs/$start_time.txt"
echo $log_name

nohup ./cluster_run.sh -o ${obs_type} -a ${aux_keys} -s -u -r > $log_name &
