#!/bin/bash
# Add NVIDIA library paths from conda installation
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

torchrun --standalone --nproc_per_node=8 train_gpt.py
