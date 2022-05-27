#!/bin/bash

out_path="models/"
link=""
reference_model="resource/reference_model.txt"

mkdir -p $out_path
while IFS='' read -r line || [[ -n "$line" ]]; do
    wget -c $link$line -O $out_path$line
done < "$reference_model"
