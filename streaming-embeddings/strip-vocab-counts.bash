#!/bin/bash

input_path=/export/projects/cmay/rcv1v2-train.text.vocab.sample
output_path=/export/projects/cmay/rcv1v2-train.text.vocab.sample.stripped

awk '{ print $2 }' < "$input_path" > "$output_path"
