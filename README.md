# EEG pretraining

Code for my thesis: "Neural Network Pretraining on Large-Scale Consumer EEG Data"

# How to start training 

```
cd src
python3.10 train.py
```

Note the config for training at configs/config_bendr.yaml. You may need to fix the paths and adjust folder with data 

On my last attempt, data was generated with preproc_slope.ipynb

# How to load chekpoints 

load checkpoints from https://drive.google.com/drive/folders/1VQMBvmu5kwJs-G-Zv9Bbq3Tj1DVL2fOi?usp=sharing

then you can load the weights:
```
with open('configs/config_bendr.yaml', 'r') as file:
      cfg = yaml.safe_load(file)
encoder = Encoder(**cfg['encoder'])
context_network = ContextNetwork(**cfg['context_network'])

encoder_sd = torch.load('encoder.pt')
context_network_sd = torch.load('context_network.pt')

encoder.load_state_dict(encoder_sd)
context_network.load_state_dict(context_network_sd)
```

For data preprocessing, refer to train.py and dataset.py
