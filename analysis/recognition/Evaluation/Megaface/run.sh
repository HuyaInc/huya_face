#!/usr/bin/env bash

DEVKIT="~/huya_face/face_datasets/megaface_data/devkit/experiments"
ALGO="model-r100-ii"
FEATURE_PATH="~/huya_face"
ROOT="~/huya_face/models"
DATA="~/huya_face/face_datasets/megaface_data/eval_set"

startTime=`date +%Y%m%d-%H:%M`
 
startTime_s=`date +%s`

python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model "$ROOT/$ALGO/model,0"

python -u remove_noises.py --algo $ALGO 

python -u gen_cleaned_megaface.py --gpu 0,1 --batch-size 96 --algo "$ALGO" --model "$ROOT/$ALGO/model,0"

cd "$DEVKIT"

python -u run_experiment.py "$FEATURE_PATH/feature_out_clean/megaface" "$FEATURE_PATH/feature_out_clean/facescrub" _"$ALGO".bin "$FEATURE_PATH/mx_results/" -s 1000000 -p ~/huya_face/face_datasets/megaface_data/devkit/templatelists/facescrub_features_list.json -d

endTime=`date +%Y%m%d-%H:%M`
 
endTime_s=`date +%s`
 
sumTime=$[ $endTime_s - $startTime_s ]
 
echo "$startTime ---> $endTime" "Totl:$sumTime seconds"
