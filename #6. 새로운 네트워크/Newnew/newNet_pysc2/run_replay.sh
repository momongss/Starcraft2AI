#!/bin/bash

i=1
while [ $i -lt 10000 ]
do
	python -m pysc2.bin.play;
	i=$((i+1))
done
