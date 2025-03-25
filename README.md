# Test-time Prompt Refinement for Text-to-Image Models

## Abstract
Text-to-image (T2I) generation models have made significant strides but still struggle with prompt sensitivity: even minor changes in prompt wording can yield inconsistent or inaccurate outputs. To address this challenge, we introduce a closed-loop, test-time prompt refinement framework that requires no additional training of the underlying T2I model, termed TIR. In our approach, each generation step is followed by a refinement step, where a pretrained multimodal language model (MLLM) analyzes the output image and the user’s prompt. The MLLM detects misalignments (e.g., missing objects, incorrect attributes) and produces a more refined and physically grounded prompt for the next round of image generation. By iteratively refining the prompt and verifying the alignment between prompt and the image, TIR corrects errors, mirroring the iterative refinement process of human artists. We demonstrate that this closed-loop strategy improves alignment and visual coherence across multiple benchmark datasets, all while maintaining plug-and-play integration with black-box T2I models.

---

## Overview
This repository implements **Test-time Image Refinement (TIR)**, a **training-free, iterative T2I pipeline** guided by a **Multimodal Large Language Model (MLLM)** to enhance the T2I generation. The workflow leverages **prompt refinement and dynamic feedback loops** to progressively improve image synthesis quality. The current implementation supports Stable Diffusion (1.4, 1.5, 2.1, XL), DALL-E, and Flux dev.1 models with customized refinement mechanisms.

---

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

## Running the TIR with Different SD Models

### Running TIR with Stable Diffusion (SD) 1.5 Model
```bash
bash scripts/run_stable_diffusion.sh
```

This script performs:
1. Prompt Extraction from `evaluation_metadata.jsonl`
2. TIR Execution (with SD 1.5 model)
3. Post-processing of the Generated Images

Within the run script, you can specify the SD version using the `--model_version` flag. Supported versions include: `1.4`, `1.5`, and `2.1`.

Output Images are saved to: `new_outputs/batch_results`

---

### Running TIR using Drawbench Dataset with SD 2.1 Model

```bash
bash scripts/run_drawbench_sd.sh
```

Workflow Steps:
1. Extract Prompts from DrawBench (`drawbench_prompts.json`)
2. Map Prompts to the Image Names (`image_prompt_mapping.json`)
3. Run TIR (with SD 2.1 model)
4. Post-process Generated Images
5. Evaluate Outputs using DrawBench Metrics

Generated Images are stored in: `drawbench/drawbench_gen_images`

Similarly, within this run script, you can specify the SD version using the `--model_version` flag. Supported versions include: `1.4`, `1.5`, and `2.1`.

---

### **Running TIR using Drawbench Dataset with DALL-E Model**

```bash
bash scripts/run_drawbench_dalle.sh
```

Workflow Steps:
1. Extract Prompts from DrawBench (`drawbench_prompts.json`)
2. Map Prompts to Expected Image Names (`image_prompt_mapping.json`)
3. Run TIR (with DALL-E model) 
4. Post-process Generated Images
5. Evaluate Outputs using DrawBench Metrics

Generated Images are stored in: `drawbench/drawbench_dalle_gen_images`

---

## **Step-by-Step Instructions (Manual Pipeline Execution)**

If you prefer to run the pipeline step-by-step without using the run scripts, follow these instructions (for running it on the drawbench dataset):

### **Step 1: Extract Prompts from DrawBench**
```bash
python src/prepare_drawbench_prompts.py  # Creates drawbench_prompts.json
```

---

### **Step 2: Generate Image-Prompt Mapping**
```bash
python src/image_prompt_mapping.py  # Creates image_prompt_mapping.json
```

---

### **Step 3: Run the TIR**
```bash
python src/main.py --model_version 2.1 --prompts_file drawbench_prompts.json --output_dir <SD_OUTPUT_DIR> --restart_steps 0 0 0 --refinement_step 99
```

---

### **Step 4: Post-process the Generated Images**
```bash

python scripts/post_process/post_process_drawbench.py --final_image final_3_refined_0_.png --output_dir <DRAWBENCH_OUTPUT_DIR> --generated_dir <SD_OUTPUT_DIR>
```

---

### **Step 5: Evaluate the Generated Images**
```bash
python drawbench/drawbench_eval.py --image_folder <DRAWBENCH_OUTPUT_DIR> --prompt_json image_prompt_mapping.json
```

---

## **Project Structure**
```
project_root/
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
│   ├── qwen_integration.py            # MLLM integration for prompt refinement
│   ├── qwen_vl_utils.py
│   ├── sdxl.py                        # SDXL with TIR implementation        
│   ├── main.py                        # Main pipeline script
│   ├── main_dalle.py                  # DALL-E pipeline script
├── filtered_prompts.json              # Extracted prompts from metadata
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
- ✅ Generated images (ex. `drawbench/drawbench_gen_images/`)
- ✅ Intermediate outputs (`new_outputs/`)

---

## Usage Notes
- If using the DALL-E pipeline, ensure `AZURE_OPENAI_ENDPOINT` is configured in your environment variables.
