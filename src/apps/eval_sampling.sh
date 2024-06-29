
#!/bin/bash

# subjects
# '987983 709551 677968 792564' 
# '782561 770352 729557 705341'
# '917255 702133 877168 679568'
# '992774 958976 765056 771354'

# '987983 709551 677968 792564 782561 770352 729557 705341' 
# '917255 702133 877168 679568 992774 958976 765056 771354'

while getopts "u:" opt; do
  IFS=' ' read -r -a UNCERTAINTY_MEASURE <<< "$OPTARG"
done

# General
POSTFIX='_no-background-budget'
CUDA_DEVICE=6
VERBOSE=true
# SUBJECTS=('987983')
SUBJECTS=('987983' '709551' '677968' '792564' '782561' '770352' '729557' '705341') # '917255' '702133' '877168' '679568' '992774' '958976' '765056' '771354')
LABELSETS=('set1' 'set2')

# User Model
INIT_VOXELS=200
REFINEMENT_VOXELS=200
NUM_INTERACTIONS=5

# Setup
GUIDANCES=('uniform' 'top_k')
SOFT_SCORES=true
BACKGROUND_BIAS=true
FEATURE='tta'
if [ "$FEATURE" == "ttd" ]; then
  DROPOUT=true
else
  DROPOUT=false
fi


for subject in "${SUBJECTS[@]}"; do
  for labelset in "${LABELSETS[@]}"; do
    for guidance in "${GUIDANCES[@]}"; do

      CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python evaluation.py \
              -cn eval_sampling \
              ++verbose="$VERBOSE" \
              ++postfix="$POSTFIX" \
              ++data.subject="$subject" \
              ++data.labelset="$labelset" \
              ++init_voxels="$INIT_VOXELS" \
              ++refinement_voxels="$REFINEMENT_VOXELS" \
              ++num_interactions=$NUM_INTERACTIONS \
              ++model.dropout=$DROPOUT \
              ++guidance="$guidance" \
              ++soft_scores="$SOFT_SCORES" \
              ++uncertainty_measure="$UNCERTAINTY_MEASURE" \
              ++background_bias="$BACKGROUND_BIAS" \
              ++feature="$FEATURE"
    done
  done
done

