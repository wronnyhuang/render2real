#!/bin/bash
# usage: sync.sh a|b  -> push code + data (+drive) to pod
cd $WORK/fast
P=$1; if [ "$P" = a ]; then unset POD; else export POD=$P; fi
IFS="|" read -r SSHARGS TARGET <<< "$(python3 pod.py sshargs)"
rsync -az -e "$SSHARGS" code/ $TARGET:/root/code/
rsync -az -e "$SSHARGS" --exclude src --exclude work --exclude qc ../data/ $TARGET:/root/data/
[ -d ../drive ] && rsync -az -e "$SSHARGS" ../drive/ $TARGET:/root/drive/
