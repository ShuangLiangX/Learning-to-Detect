# Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models

Official implementation of **“Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models.”**

This repository contains the data-processing, hidden-state extraction, classifier training, safety-pattern auto-encoder, and evaluation code used in the project.

This release provides the LLaVA-v1.6-Vicuna-7B implementation. Adapting the pipeline to Qwen2.5-VL or CogVLM requires model-specific input processing and separately trained detectors.

## Contents

- [Models](#models)
- [Repository Structure](#repository-structure)
- [Detection Pipeline](#detection-pipeline)
- [Datasets](#datasets)

## Models

The experiments use the following base models:

- [LLaVA-v1.6-Vicuna-7B](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b), a large vision-language model based on Vicuna.
- [Llama Guard 3 8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B), a safety guardrail model used to assess generated responses.

Download the model weights separately and place them under:

```text
asset/weights/
```

## Repository Structure

```text
.
├── asset/
│   ├── advbench/                 # AdvBench images
│   ├── harmbench/                # HarmBench DirectRequest images
│   ├── GQA/                      # GQA images
│   ├── HiddenStates/             # Extracted hidden states
│   └── weights/                  # Model weights (not included)
├── Benchmarks/                   # Evaluation benchmark metadata
├── vicuna/
│   ├── instructions/
│   │   ├── advbench.json
│   │   ├── GQA.json
│   │   └── harmbench.json        # HarmBench DirectRequest metadata
│   ├── qa.py
│   ├── qa-baseline.py
│   └── train.py
├── autoencoder.py
├── llama3_guard.py
└── test.py
```

## Detection Pipeline

Some scripts use paths relative to their own working directory. Run the commands from the directories shown below.

### 1. Query the vision-language model

Query the model on the unsafe source data (AdvBench) and safe source data (GQA):

```bash
cd vicuna
python qa.py --dataset advbench
python qa.py --dataset GQA
cd ..
```

The HarmBench DirectRequest data can be queried with the same interface:

```bash
cd vicuna
python qa.py --dataset harmbench
cd ..
```

### 2. Assess and split model responses

Assess the generated AdvBench responses with Llama Guard 3:

```bash
python llama3_guard.py --file vicuna/instructions/advbench.json
```

Create the AdvBench and GQA training/test splits:

```bash
cd vicuna/instructions
python process.py
cd ../..
```

### 3. Extract hidden states

Extract hidden states for the evaluation benchmarks currently configured in `vicuna/qa-baseline.py`:

```bash
cd vicuna
python qa-baseline.py
cd ..
```

### 4. Train and test the MSCAV classifiers

```bash
cd vicuna
python train.py --train
python train.py --test
cd ..
```

### 5. Train the Safety Pattern Auto-Encoder (SPAE)

```bash
python autoencoder.py
```

### 6. Evaluate detection performance

```bash
python test.py
```

## Datasets

### Training and appendix data

| Dataset | Role | Metadata | Images |
| --- | --- | --- | --- |
| AdvBench | Unsafe source data | `vicuna/instructions/advbench.json` | `asset/advbench/` |
| GQA | Safe source data | `vicuna/instructions/GQA.json` | `asset/GQA/` |
| HarmBench (DirectRequest) | Unsafe-source experiment reported in the appendix | `vicuna/instructions/harmbench.json` | `asset/harmbench/` |

The included HarmBench DirectRequest subset contains **320** image–request pairs. Image paths in `harmbench.json` are repository-relative and follow the same layout convention as AdvBench.

### Evaluation benchmarks

Metadata for the included evaluation sets is stored in `Benchmarks/`, including FC, HADES, JOOD, MML-m, and SafetyBench variants.

Each included attack JSON contains 400 samples. The evaluation script uses all supplied hidden states without additional sampling.

Download the corresponding attack images and matching benchmark metadata from **[ModelScope: detecpolo/Learning-to-Detect](https://modelscope.cn/datasets/detecpolo/Learning-to-Detect)**. The package covers all six attack sets (400 records per attack, with 1,602 distinct images; VAJM and UMK each share one attack image).

Copy the downloaded `asset/` and `Benchmarks/` directories into the repository root, preserving their directory structure. The package does not include the `mm-vet` safe evaluation data, model weights, or extracted hidden states.

As described in Section 4.1 of the paper, FC-Attack, JOOD, MML, VAJM, and UMK inputs are constructed by applying the corresponding attack methods to unsafe inputs from MM-SafetyBench (13 safety-related scenarios). HADES uses its own released dataset (five categories). Appendix B describes the expanded-set evaluation; the downloadable package above contains the 400-sample attack sets.


For the safe evaluation set, provide `Benchmarks/mm-vet.json` containing 400 safe samples, with `question` and `image` fields for each sample, and the corresponding images. This metadata and the safe hidden states are not included. The extraction script writes `asset/HiddenStates/mm-vet_answer.pth`, which is shared across attack evaluations.

VAJM and UMK use the included `asset/adversarial_images/vajm-vicuna.bmp` and `asset/adversarial_images/umk-vicuna.bmp`, respectively. The extractor overrides the clean-image paths in their metadata and appends the provided UMK suffix.

FC metadata expects the flowchart images under `asset/adversarial_images/FC_Attack-main/data_flowchart/vertical/generated/`. These images are included in the ModelScope package linked above. Metadata image paths are resolved relative to the repository root.

To extract only selected benchmarks from the `vicuna` directory:

```bash
python qa-baseline.py --datasets FC SafetyBench-vajm SafetyBench-umk
```


External dataset resources:

- [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench)
- [HADES](https://github.com/AoiDragon/HADES)

Place any separately downloaded dataset assets under `asset/` and keep the image paths in the corresponding JSON files consistent with the local directory layout.
