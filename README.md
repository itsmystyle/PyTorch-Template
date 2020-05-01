# PyTorch Template Project
PyTorch deep learning project made easy.

## Folder Structure
PyTorch-Template/
│
├── scripts/ - main script
│   ├── train.py - main script to start training
│   └── test.py - evaluation of trained model
│
├── module/ - main module source scode
│   ├── dataset/ - anything about data goes here
│   │   ├── factory.py - how to prepare dataset goes here
│   │   └── mnist.py - dataset module
│   │
│   ├── loss/ - anything about loss function goes here
│   │   ├── factory.py - how to prepare loss function goes here
│   │   └── label_smoothing.py - example loss function
│   │
│   ├── metrics/ - anything about metrics goes here
│   │   ├── factory.py - how to prepare metrics goes here
│   │   └── accuracy.py - example accuracy metric
│   │
│   ├── model/ - anything about model goes here
│   │   ├── factory.py - how to prepare model goes here
│   │   └── vgg.py - example model architecture
│   │
│   ├── optimizer/ - anything about optimizer goes here
│   │   ├── factory.py - how to prepare optimizer goes here
│   │   └── multi_optimizer.py - example optimizer module
│   │
│   ├── trainer/ - anything about trainer goes here
│   │   ├── trainer_base.py - base trainer blue print
│   │   └── trainer.py
│   │
│   ├── argparser.py - anything about arguments goes here
│   │
│   └── utils.py - anything about small utility functions go here
│
├── .editorconfig
├── .gitignore
├── requirements.txt
│
└── saved/
    ├── models/ - trained models are saved here
    └── train_logs/ - default logdir for tensorboard and logging output

## Usage
``` bash
    python -m scripts.train
```
