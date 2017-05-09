#!/bin/bash

input_path=/export/projects/cmay/rcv1v2-train.text.split
output_path=/export/projects/cmay/rcv1v2-train.text.vocab

sed 's/ /\n/g' < "$input_path" | sort | uniq -c > "$output_path"
