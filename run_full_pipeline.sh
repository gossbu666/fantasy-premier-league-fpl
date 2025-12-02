#!/bin/bash
echo "========================================"
echo "FULL DATA PIPELINE - 755 PLAYERS"
echo "Started at: $(date)"
echo "========================================"

echo "\n[1/3] Data Collection (60-90 minutes)..."
python3 main_pipeline.py

echo "\n[2/3] Feature Engineering (5 minutes)..."
python3 feature_pipeline.py

echo "\n[3/3] Model Training (10-15 minutes)..."
python3 train_models.py

echo "\n========================================"
echo "PIPELINE COMPLETE!"
echo "Finished at: $(date)"
echo "========================================"

