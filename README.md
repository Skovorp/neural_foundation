## Neural Network Pretraining on Large-Scale Consumer EEG Data

This project explores self-supervised learning for EEG by adapting the wav2vec 2.0 architecture. It replicates and improves upon the BENDR paper, applying masked autoencoding techniques to EEG data. Unlike previous studies that struggled with heterogeneous datasets, this work focuses on a single-device dataset consisting of 5k hours of proprietary unlabeled EEG recordings. The goal is to learn general-purpose EEG representations that can be fine-tuned for various downstream tasks.

## Repository Structure
```
src/
├── train.py # Training script
├── model/
│ ├── bendr.py # Model architecture implementation
├── loss/
│ ├── loss_bendr.py # Loss function implementation
├── configs/
│ ├── od_config_bendr.yaml # Final configuration used for training
```
