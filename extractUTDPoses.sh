#!/bin/bash

for filename in selectedGesturesUTD/*.avi; do
    echo $filename
    python3 PoseFromVideo.py --video=$filename --outcsv="$filename".csv --dbgimgs=false
done
