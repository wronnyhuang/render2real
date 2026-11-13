#!/bin/bash
cd $WORK/fast
while true; do
  done_flag=0; [ -f ../data/DONE ] && done_flag=1
  for P in $PODS; do timeout 600 ./sync.sh $P && echo "$(date +%T) synced $P $(python3 -c "import json;print(len(json.load(open('../data/manifest.json'))['clips']))")"; done
  [ $done_flag = 1 ] && { echo FINAL_SYNC_DONE; exit 0; }
  sleep 420
done
