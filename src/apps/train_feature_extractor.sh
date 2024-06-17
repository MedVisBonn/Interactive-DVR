
#!/bin/bash

# General
NAME='ae_feature-extractor_dropout'
LOG=true
CUDA_DEVICE=5
TRAIN=true

# Model
ENCODER='zero'
DROPOUT=false
N_EPOCHS=20
PATIENCE=2

# ++subjects="$SUBJECTS" \
if [ "$TRAIN" = true ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_feature_extractor.py \
        -cn ae_training \
        ++name=$NAME \
        ++log="$LOG" \
        ++model.encoder="$ENCODER" \
        ++model.dropout="$DROPOUT" \
        ++n_epochs="$N_EPOCHS" \
        ++patience="$PATIENCE"
fi