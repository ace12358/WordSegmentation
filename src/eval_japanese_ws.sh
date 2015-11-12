#!/bin/sh
# $ bash  eval_japanese_ws.sh ref_file sys_file
ref_file=$1
sys_file=$2

echo "ref_file : ${ref_file}"
echo "sys_file : ${sys_file}"

python pre_treatment.py ${ref_file} ${sys_file} |perl conlleval.pl -d "\t"

