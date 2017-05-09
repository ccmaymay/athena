#!/bin/bash

set -e

input_path=word2vec/word2vec/trunk/text8
output_dir=/export/projects/cmay
output_stem=text8

[ -f streaming-embeddings/text8-sim.bash ]

mkdir -p $output_dir

pairs() {
    join <(sed 's/^/1 /' $1) <(sed 's/^/1 /' $2) | sed 's/^1 //;s/ /\t/' > $3
}

echo 'computing word pairs ...'
build/lib/naive-lm-train-raw $input_path $output_dir/${output_stem}.lm
build/lib/lm-print $output_dir/${output_stem}.lm $output_dir/${output_stem}.1-end
head -n 100 < $output_dir/${output_stem}.1-end > $output_dir/${output_stem}.1-100
head -n 900 < $output_dir/${output_stem}.1-end | tail -n 100 > $output_dir/${output_stem}.801-900
head -n 6500 < $output_dir/${output_stem}.1-end | tail -n 100 > $output_dir/${output_stem}.6401-6500
python streaming-embeddings/unique-pairs.py $output_dir/${output_stem}.1-100 $output_dir/${output_stem}.1-100.1-100
python streaming-embeddings/unique-pairs.py $output_dir/${output_stem}.801-900 $output_dir/${output_stem}.801-900.801-900
python streaming-embeddings/unique-pairs.py $output_dir/${output_stem}.6401-6500 $output_dir/${output_stem}.6401-6500.6401-6500
pairs $output_dir/${output_stem}.1-100 $output_dir/${output_stem}.801-900 $output_dir/${output_stem}.1-100.801-900
pairs $output_dir/${output_stem}.1-100 $output_dir/${output_stem}.6401-6500 $output_dir/${output_stem}.1-100.6401-6500
pairs $output_dir/${output_stem}.801-900 $output_dir/${output_stem}.6401-6500 $output_dir/${output_stem}.801-900.6401-6500

suffixes='1-100.1-100 1-100.801-900 1-100.6401-6500 801-900.801-900 801-900.6401-6500 6401-6500.6401-6500'

rm -f $output_dir/${output_stem}.plot-sim.log

echo 'training word2vec model ...'
build/lib/word2vec-train-raw $input_path $output_dir/${output_stem}.w2v
echo 'computing similarity ...'
for suffix in $suffixes
do
    build/lib/sgns-model-print-similarity $output_dir/${output_stem}.w2v $output_dir/${output_stem}.${suffix} $output_dir/${output_stem}.${suffix}.sim.w2v
done

echo 'training space-saving word2vec model ...'
build/lib/spacesaving-word2vec-train-raw $input_path $output_dir/${output_stem}.ssw2v
echo 'computing similarity ...'
for suffix in $suffixes
do
    build/lib/sgns-model-print-similarity $output_dir/${output_stem}.ssw2v $output_dir/${output_stem}.${suffix} $output_dir/${output_stem}.${suffix}.sim.ssw2v
done
echo 'plotting ...'
for suffix in $suffixes
do
    python streaming-embeddings/plot-sim.py --sim-name ssw2v --output-path ssw2v_vs_word2vec.pdf $output_dir/${output_stem}.${suffix}.sim.w2v $output_dir/${output_stem}.${suffix}.sim.ssw2v | tee -a $output_dir/${output_stem}.plot-sim.log
done

for i in {0..3}
do
    echo "training word2vec model $i ..."
    build/lib/word2vec-train-raw $input_path $output_dir/${output_stem}.w2v.$i
    echo 'computing similarity ...'
    for suffix in $suffixes
    do
        build/lib/sgns-model-print-similarity $output_dir/${output_stem}.w2v.$i $output_dir/${output_stem}.${suffix} $output_dir/${output_stem}.${suffix}.sim.w2v.$i
    done
    echo 'plotting ...'
    for suffix in $suffixes
    do
        python streaming-embeddings/plot-sim.py --sim-name word2vec.$i --output-path word2vec_${i}_vs_word2vec.pdf $output_dir/${output_stem}.${suffix}.sim.w2v $output_dir/${output_stem}.${suffix}.sim.w2v.$i | tee -a $output_dir/${output_stem}.plot-sim.log
    done
done

echo 'done'
