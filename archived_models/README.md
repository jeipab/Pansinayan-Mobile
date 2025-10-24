# Archived Classification Models

This directory contains the original PyTorch classification models that were used for the previous sign language recognition system.

## Models

- **MediaPipeGRU_best.pt** (286 KB) - GRU-based classification model
- **SignTransformer_best.pt** (39.7 MB) - Transformer-based classification model

## Purpose

These models were trained for classification tasks with output shape `[1, 105]` (105 gloss classes). They have been archived as we transition to CTC-based continuous recognition models with output shape `[1, 300, 106]` (105 glosses + 1 blank token).

## Usage

These models can be used as:

- Reference for model architecture
- Starting point for CTC model training
- Fallback for classification-only tasks

## Note

The new CTC pipeline will require retraining these models with CTC loss and appropriate output shapes for continuous recognition.
