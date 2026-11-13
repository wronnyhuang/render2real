#!/bin/bash
cd $WORK/turbo0
IFS="|" read -r SSHARGS TARGET <<< "$(python3 pod.py sshargs)"
rsync -az -e "$SSHARGS" code/ $TARGET:/root/code/ && rsync -az -e "$SSHARGS" jobs.json setup_pod.sh $TARGET:/root/
rsync -az -e "$SSHARGS" $WORK/fast/weights/turbo_lora.pt $TARGET:/root/
rsync -az -e "$SSHARGS" --files-from=files.txt $WORK/ $TARGET:/root/
