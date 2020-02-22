#!/bin/bash

azure_config=azure_config.json
if [ ! -f ${azure_config} ]; then
    echo "Cannot find $azure_config"
    exit 1
fi

parallel=true
command -v pdsh
if [ $? != 0 ]; then
    echo "Installing pdsh will allow for the docker pull to be done in parallel across the cluster. See: 'apt-get install pdsh'"
    parallel=false
fi

ssh_key=`cat ${azure_config} | jq .ssh_private_key | sed 's/"//g'`
if [ $ssh_key == "null" ]; then echo 'missing ssh_private_key in config'; exit 1; fi
num_vms=`cat ${azure_config} | jq .num_vms`
if [ $num_vms == "null" ]; then echo 'missing num_vms in config'; exit 1; fi
location=`cat ${azure_config} | jq .location | sed 's/"//g'`

args="-i ${ssh_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
username=deepspeed
resource_group=deepspeed_rg_$location

# JKK: need to get this to pull or scp Hazy code, too ...
# JKK: i.e. the GAN script for this test, u.s.w. ...

update_script="
docker pull deepspeed/deepspeed:latest;
ln -s workdir/DeepSpeed/azure/attach.sh attach.sh;
cd workdir/DeepSpeed;
git pull;
git submodule update --init --recursive;
bash azure/start_container.sh
"

# JKK: unfortunatrly, the pdsh command doesn't seem to work, so need to investigate
# JKK: this; however, the rest of it is behaving itself ...

#if [ $parallel == true ]; then
#    echo "parallel docker pull"
#    hosts=""
#    for node_id in {0..1}; do
#        addr=`az vm list-ip-addresses  -g ${resource_group} | jq .[${node_id}].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
#        hosts="${addr},${hosts}"
#    done
#    echo "About to execute command: ${args} pdsh -w $hosts -l ${username} $update_script"
#    PDSH_SSH_ARGS_APPEND=${args} pdsh -w $hosts -l ${username} $update_script
#else


echo "sequential docker pull"
for node_id in `seq 0 $((num_vms - 1))`; do
  ip_addr=`az vm list-ip-addresses  -g ${resource_group} | jq .[${node_id}].virtualMachine.network.publicIpAddresses[0].ipAddress | sed 's/"//g'`
  addr=${username}@${ip_addr}
  echo "about to run: ssh ${args} $addr $update_script"
  ssh ${args} $addr $update_script
done


#fi
