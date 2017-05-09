#!/bin/bash

input_path=/export/common/kduh/data/rcv1/processed/rcv1v2-train.text
output_path=/export/projects/cmay/rcv1v2-train.text.split

sed 's/ \([.!?]\) / \1\n/g;s/^ *//;s/ *$//' < "$input_path" | \
    sed '/^$/d' > "$output_path"
