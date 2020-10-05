#!/bin/bash

DIR='/home/evahu/Documents/Master/Master_Dissertation/InflDerivMorph/data/FINAL/'

for D in $DIR*; do
	for subd in $D/*; do
		python3 meta_data.py $subd nokeyword
	done
done 