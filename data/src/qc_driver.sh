#!/bin/bash
# incremental QC: each finished pair (meta.json present) gets qc/qc_<id>.json; 6 at a time on cores 64-69 pairs
D=$WORK/data; PY=$WORK/env/mp/bin/python
cd $D
while true; do
  todo=$(for m in pairs/M*/meta.json; do id=$(basename $(dirname $m)); [ -f qc/qc_$id.json ] || [ -f qc/qc_$id.run ] || echo $id; done)
  n=$(ls qc/*.run 2>/dev/null | wc -l)
  for id in $todo; do
    [ $n -ge 8 ] && break
    touch qc/qc_$id.run
    (nice -n 15 taskset -c 62-69 $PY -I src/qc.py $id > qc/log_$id.txt 2>&1; rm -f qc/qc_$id.run) &
    n=$((n+1))
  done
  if ! kill -0 $(cat logs/render_driver.pid) 2>/dev/null && [ -z "$todo" ] && [ $(ls qc/*.run 2>/dev/null | wc -l) -eq 0 ]; then break; fi
  sleep 10
done
echo qc done
