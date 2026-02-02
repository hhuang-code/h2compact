#!/bin/bash

source ~/.bashrc
conda activate wham


root="./WHAM/extracted_data/ex_data"                 # default: current directory

find "$root" -type f -iname '*.mp4' -print0 |
while IFS= read -r -d '' video; do
    folder=$(dirname "$video")     # assign containing directory
    echo "video = $video"
    echo "folder = $folder"
    echo "----------"

    python demo.py --video $video --output_pth $folder --visualize --run_smplify
done