#!/bin/bash
set -e
SRC=/data/skurchev/workspace/datasets/hf_release_cosmos3_140_200
echo "== batch_140_sam3d_objects samples =="
ls "$SRC/batch_140_sam3d_objects" | grep -v pycache | head -5
echo "== batch_200_oreo_grasp samples =="
ls "$SRC/batch_200_oreo_grasp" | grep -v pycache | head -5
echo "== resolution: first sample_ dir from batch_140 =="
D140=$(ls "$SRC/batch_140_sam3d_objects" | grep -v pycache | grep -v '\.py$' | head -1)
echo "dir=$D140"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$SRC/batch_140_sam3d_objects/$D140/video_L.mp4" || echo "no video_L.mp4 here"
echo "== resolution: first sample dir from batch_200 =="
D200=$(ls "$SRC/batch_200_oreo_grasp" | grep -v pycache | head -1)
echo "dir=$D200"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$SRC/batch_200_oreo_grasp/$D200/generated_video.mp4" || echo "no generated_video.mp4 here"
echo "== resolution: several more from batch_140 =="
for d in $(ls "$SRC/batch_140_sam3d_objects" | grep -v pycache | grep -v '\.py$' | head -6); do
  f="$SRC/batch_140_sam3d_objects/$d/video_L.mp4"
  if [ -f "$f" ]; then
    printf "%s: " "$d"
    ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$f"
  fi
done
echo "== resolution: several more from batch_200 =="
for d in $(ls "$SRC/batch_200_oreo_grasp" | grep -v pycache | head -6); do
  f="$SRC/batch_200_oreo_grasp/$d/generated_video.mp4"
  if [ -f "$f" ]; then
    printf "%s: " "$d"
    ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$f"
  fi
done
