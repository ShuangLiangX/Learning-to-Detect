# Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models

This repository provides the official implementation of **"Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models"**.

## Content
- [Base model](#base-model)
- [Jailbreak Attack Detection](#jailbreak-attack-detection)

- [Dataset](#dataset)

## Base model 

Our method uses the following two base models:

[LLaVA-v1.6-Vicuna](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) is a powerful vision-language model that combines a visual encoder with the Vicuna language model to process multimodal inputs and generate natural language responses.

[LlamaGuard3](https://huggingface.co/meta-llama/Llama-Guard-3-8B) is a safety guardrail model developed by Meta AI, specifically designed to detect and prevent harmful content generation and effectively identify potentially unsafe requests and responses.

Please download the model weights and place them in the `code/asset/weights` directory.

## Jailbreak Attack Detection

### 1. Data Processing and Hidden State Extraction  
*(Optional, since the processed data and extracted states are already preserved in the repository.)*

#### Query the model on $I^-$ (AdvBench) and $I^+$ (GQA)
```
python code/vicuna/qa.py --file code/vicuna/instructions/advbench.json
python code/vicuna/qa.py --file code/vicuna/instructions/GQA.json
```
#### Assess model responses and split into training/testing datasets
```
    python code/llama3_guard.py --file code/vicuna/instructions/advbench.json
    python code/vicuna/instructions/process.py 
```

#### Extract hidden states for LoD training and benchmark evaluation
```
    python code/vicuna/qa-baseline.py 
```

### 2. Train and Test the MSCAV classifiers
####    Train and test classifiers
```
    python code/vicuna/train.py --train
    python code/vicuna/train.py --test
```

### 3. Train the Safety Pattern Auto-Encoder (SPAE)

```
    python code/autoencoder.py
```

### 4.  Evaluate Detection Performance
```
    python code/test.py
```

## Dataset
| Dataset | Details |
|------|-----|
| [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench)  |   |
| [HADES](https://github.com/AoiDragon/HADES)   |  |


Please download the datasets and place them in the `code/asset` directory.
