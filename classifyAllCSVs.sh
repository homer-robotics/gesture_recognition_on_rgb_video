#!/bin/bash

for filename in poseCSVs/*.csv; do
    echo $filename
    python3 DTWClassifier.py -cp=$filename -vt=0.10 -s=3
    python3 showConfusionMatrix.py
done
