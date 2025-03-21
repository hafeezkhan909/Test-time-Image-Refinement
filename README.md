# Test-time Prompt Refinement for Text-to-Image Models

## Abstract
Text-to-image (T2I) generation models have made significant strides but still struggle with prompt sensitivity: even minor changes in prompt wording can yield inconsistent or inaccurate outputs. To address this challenge, we introduce a closed-loop, test-time prompt refinement framework that requires no additional training of the underlying T2I model, termed TIR. In our approach, each generation step is followed by a refinement step, where a pretrained multimodal language model (MLLM) analyzes the output image and the user’s prompt. The MLLM detects misalignments (e.g., missing objects, incorrect attributes) and produces a more refined and physically grounded prompt for the next round of image generation. By iteratively refining the prompt and verifying the alignment between prompt and the image, TIR corrects errors, mirroring the iterative refinement process of human artists. We demonstrate that this closed-loop strategy improves alignment and visual coherence across multiple benchmark datasets, all while maintaining plug-and play integration with black-box T2I models.

## Overview
This repository implements a **training-free, iterative diffusion model pipeline** guided by a **Mulitmodal Large Language Model (MLLM)** to enhance text-to-image generation. The workflow leverages **prompt refinement and dynamic feedback loops** to progressively improve image synthesis quality. The approach supports Stable Diffusion (1.4, 1.5, 2.1, XL), DALL-E, and Flux dev.1 models with customized refinement mechanisms.

## Installation & Setup
### 1️⃣ Environment Setup
Run the setup script to create and activate a virtual environment, and install dependencies:

```bash
bash scripts/setup.sh
```

---

### 2️⃣ Configure Environment Variables
Load environment settings for the pipeline:
```bash
source scripts/config.sh
```
This sets:
- `PYTHONPATH` → Source directory
- `PROMPTS_FILE` → Filtered prompts file
- `OUTPUT_DIR` → Output directory

---

## Running the Pipelines
### Running Stable Diffusion Pipeline (1.5)
The script runs the complete Stable Diffusion 1.5 pipeline with prompt refinement:
```bash
bash scripts/run_stable_diffusion.sh
```

This script performs:
1. Prompt Extraction from `evaluation_metadata.jsonl`
2. Diffusion Pipeline Execution (Stable Diffusion 1.5)
3. Post-processing of Generated Images

Output Images are saved to: `new_outputs/batch_results`

---

### Running DrawBench Evaluation with Stable Diffusion (2.1)
This pipeline evaluates prompts using DrawBench:
```bash
bash scripts/run_drawbench_sd.sh
```

Workflow Steps:
1. Extract Prompts from DrawBench (`drawbench_prompts.json`)
2. Map Prompts to the Image Names (`image_prompt_mapping.json`)
3. Run Diffusion Pipeline for Image Generation (Stable Diffusion 2.1)
4. Post-process Generated Images
5. Evaluate Outputs using DrawBench Metrics

Generated Images are stored in: `drawbench/drawbench_gen_images`

---

### **Running DrawBench Evaluation with DALL-E**
This pipeline runs the evaluation using the DALL-E model:
```bash
bash scripts/run_drawbench_dalle.sh
```

Workflow Steps:
1. Extract Prompts from DrawBench (`drawbench_prompts.json`)
2. Map Prompts to Expected Image Names (`image_prompt_mapping.json`)
3. Run DALL-E Pipeline for Image Generation (`main_dalle.py`)
4. Post-process Generated Images
5. Evaluate Outputs using DrawBench Metrics

Generated Images are stored in: `drawbench/drawbench_dalle_gen_images`

---

## **Project Structure**
```
project_root/
├── generated_images/                  # Final processed images
├── new_outputs/                       # Intermediate results
├── filtered_prompts/                  # Extracted prompts from metadata
├── drawbench/                         # DrawBench-related outputs and evaluations
├── evaluation_metadata.jsonl          # Input metadata file
├── scripts/                           # Scripts for running pipelines
│   ├── extract/                       # Prompt extraction scripts
│   ├── post_process/                  # Post-processing scripts
│   ├── run_stable_diffusion.sh        # Pipeline for Stable Diffusion
│   ├── run_drawbench_sd.sh            # Pipeline for DrawBench with Stable Diffusion
│   ├── run_drawbench_dalle.sh         # Pipeline for DrawBench with DALL-E
│   ├── setup.sh                       # Environment setup
│   ├── config.sh                      # Environment configuration
│   ├── clean.sh                       # Cleanup script
├── src/                               # Source code
│   ├── diffusion/                     # Diffusion model implementation
│   ├── qwen_integration.py            # LLM integration for prompt refinement
│   ├── main.py                        # Main pipeline script
│   ├── main_dalle.py                  # DALL-E pipeline script
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

---

## Cleaning Up
To clean up all generated files and reset the environment, run:
```bash
bash scripts/clean.sh
```
This will delete:
- ✅ Virtual environment (`diff/`)
- ✅ Generated images (`generated_images/` and `drawbench/drawbench_gen_images/`)
- ✅ Intermediate outputs (`new_outputs/`)

---

## Usage Notes
- If using the DALL-E pipeline, ensure `AZURE_OPENAI_ENDPOINT` is configured in your environment variables.