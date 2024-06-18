
#!/bin/bash

# subjects
# '987983 709551 677968 792564' 
# '782561 770352 729557 705341'
# '917255 702133 877168 679568'
# '992774 958976 765056 771354'

# '987983 709551 677968 792564 782561 770352 729557 705341' 
# '917255 702133 877168 679568 992774 958976 765056 771354'

while getopts "s:" opt; do
  IFS=' ' read -r -a SUBJECTS <<< "$OPTARG"
done

# General
POSTFIX=''
CUDA_DEVICE=6
VERBOSE=true
# SUBJECTS=('987983' '709551' '677968' '792564' '782561' '770352' '729557' '705341' '917255' '702133' '877168' '679568' '992774' '958976' '765056' '771354')
LABELSETS=('set1' 'set2')

# User Model
INIT_VOXELS=200
REFINEMENT_VOXELS=200
NUM_INTERACTIONS=10

# Setup
UNCERTAINTY_MEASURE='entropy'
BACKGROUND_BIAS=('true' 'false')
FEATURE='default'

if [ "$FEATURE" == "ttd" ]; then
  DROPOUT=true
else
  DROPOUT=false
fi



for subject in "${SUBJECTS[@]}"; do
  for labelset in "${LABELSETS[@]}"; do
    for bb in "${BACKGROUND_BIAS[@]}"; do

      CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python evaluation.py \
          -cn eval_background_bias \
          ++verbose="$VERBOSE" \
          ++postfix="$POSTFIX" \
          ++data.subject="$subject" \
          ++data.labelset="$labelset" \
          ++init_voxels="$INIT_VOXELS" \
          ++refinement_voxels="$REFINEMENT_VOXELS" \
          ++num_interactions=$NUM_INTERACTIONS \
          ++model.dropout=$DROPOUT \
          ++uncertainty_measure="$UNCERTAINTY_MEASURE" \
          ++background_bias="$bb" \
          ++feature="$FEATURE"
    done
  done
done

