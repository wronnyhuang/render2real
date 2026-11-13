#!/bin/bash
# grid.sh <clip> <out.png> <variant...> : rows = t=1,3,5s ; cols = render, variants..., real (if exists)
cd $WORK/wan
clip=$1; o=$2; shift 2
base=${clip%_trainsample}; src=../data/pairs/$base; [ -d $src ] || src=""
r=../data/pairs/$base/render.mp4; [ -f $r ] || r=../drive/$clip.mp4
ins=(); n=0
for t in 1 3 5; do
  ins+=(-ss $t -i $r); n=$((n+1))
  for v in "$@"; do ins+=(-ss $t -i out/$v/$clip.mp4); n=$((n+1)); done
  [ -n "$src" ] && { ins+=(-ss $t -i $src/real.mp4); n=$((n+1)); }
done
cols=$(( n / 3 ))
f=""; for i in $(seq 0 $((n-1))); do f+="[$i:v]trim=end_frame=1,scale=256:256,setpts=PTS-STARTPTS[v$i];"; done
for row in 0 1 2; do s=""; for c in $(seq 0 $((cols-1))); do s+="[v$((row*cols+c))]"; done; f+="${s}hstack=$cols[r$row];"; done
f+="[r0][r1][r2]vstack=3"
ffmpeg -v error -y "${ins[@]}" -filter_complex "$f" -frames:v 1 $o
