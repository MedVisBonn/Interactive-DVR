
#!/bin/bash

# General
CUDA_DEVICE=5
VERBOSE=true
SUBJECTS=2

# User Model
INIT_VOXELS=200
REFINEMENT_VOXELS=200
NUM_INTERACTIONS=10

# Setup
UNCERTAINTY_MEASURE='entropy'
BACKGROUND_BIAS=true
FEATURE='tta'


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python evaluation.py \
    -cn eval \
    ++verbose="$VERBOSE" \
    ++init_voxels="$INIT_VOXELS" \
    ++refinement_voxels="$REFINEMENT_VOXELS" \
    ++num_interactions=$NUM_INTERACTIONS \
    ++uncertainty_measure="$UNCERTAINTY_MEASURE" \
    ++background_bias="$BACKGROUND_BIAS" \
    ++feature="$FEATURE"