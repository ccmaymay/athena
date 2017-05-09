#!/bin/bash

input_path=/export/projects/cmay/rcv1v2-train.text.vocab
output_path=/export/projects/cmay/rcv1v2-train.text.vocab.sample

awk 'NR % 100 == 0 { print $0 }' < "$input_path" > "$output_path"
