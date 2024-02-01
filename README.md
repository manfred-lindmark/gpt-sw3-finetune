# Fine-tune GPT-SW3 using quantized LORA

This repository contains minimal code for demonstrating fine-tuning of the GPT-SW3 language models using the QLORA method. The code is written for use with a Nvidia GPU and tested on Windows. The training setup should work with any of the model sizes. With 24 GB of VRAM you can train the 6.7B model with a batch size of 2. To train with less memory you can reduce batch size, enable gradient checkpointing, reduce LORA rank and layers, or simply swap to a smaller base GPT model. 

## Setting up Python requirements

Anaconda is a recommended way of installing the required Python packages.
The command below will create an environment with most requirements.

```bash
conda create --name gpt-sw3-finetune python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 transformers peft sentencepiece huggingface_hub -c pytorch -c nvidia -c huggingface -c conda-forge
```

Additionally you need "bitsandbytes" which does not have official Windows support. There are however precompiled versions available. Make sure your Anaconda environment is activated and run the following to install bitsandbytes on Windows. This is the latest version as of January 2024. On linux it can be installed with pip.

```bash
conda activate gpt-sw3-finetune
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

## Dataset (text corpus) for training
This example is written for smaller datasets in a single .txt file.
You can get a ~12 MB sample Swedish dataset containing horror stories from the "creepypodden" podcast by running get_dataset.py. This requires a few more python packages.
```bash
conda install anaconda::beautifulsoup4 anaconda::html5lib
python get_dataset.py
```

## Inference: just generating text

## Training

## Troubleshooting Python environment

New versions of the required Python packages are frequently released so if there are any problems you should follow the installation guides for each of them to check their compatibility.

## Troubleshooting training

Learning rate is a good first step to try increasing/decreasing.