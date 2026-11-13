#!/bin/bash
# usage: rs.sh <src> <dst>  (use R: prefix for remote paths)
E="ssh -i $SSH_KEY -o BatchMode=yes -o UserKnownHostsFile=$WORK/wan/known_hosts -o ControlMaster=auto -o ControlPersist=1800 -o ControlPath=$WORK/wan/cm-%C -p ${PP:-40152}"
a=${1/#R:/root@${PH:?set PH=pod-ip}:}; b=${2/#R:/root@${PH:?set PH=pod-ip}:}
exec rsync -az --no-owner --no-group -e "$E" "$a" "$b"
