#!/bin/bash
# contact sheet: rows = 4 clips (frame 60), cols = render, base_s025, base_s050, base_s075, base_asft, turbo_lora
set -e
cd $WORK/turbo0
R=$WORK
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
W=256
T=sheet_tmp; rm -rf $T; mkdir -p $T
CLIPS="M023_P0370 M099_P1067 juno_01 muse_01"
COLS="render base_s025 base_s050 base_s075 base_asft turbo_lora"
src() { # clip col
  if [ $2 = render ]; then case $1 in muse*|juno*) echo $R/drive/$1.mp4;; *) echo $R/data/pairs/${1%_trainsample}/render.mp4;; esac
  else echo $R/fast/out/$2/$1.mp4; fi; }
rows=()
for c in $CLIPS; do
  ins=(); for col in $COLS; do
    ffmpeg -v error -y -i "$(src $c $col)" -vf "select=eq(n\,60),scale=$W:$W" -frames:v 1 $T/${c}_$col.png
    ins+=(-i $T/${c}_$col.png); done
  ffmpeg -v error -y "${ins[@]}" -filter_complex "hstack=6,drawtext=fontfile=$F:text=$c:x=6:y=6:fontsize=16:fontcolor=yellow:box=1:boxcolor=black@0.6" $T/row_$c.png
  rows+=(-i $T/row_$c.png)
done
hdr=""; i=0; for col in $COLS; do hdr="$hdr,drawtext=fontfile=$F:text=$col:x=$((i*W+8)):y=10:fontsize=20:fontcolor=white"; i=$((i+1)); done
ffmpeg -v error -y -f lavfi -i "color=c=black:s=$((6*W))x40:d=1" -frames:v 1 -vf "${hdr#,}" $T/hdr.png
ffmpeg -v error -y -i $T/hdr.png "${rows[@]}" -filter_complex "vstack=5" -q:v 3 sheet.jpg
rm -rf $T; echo SHEET_OK
