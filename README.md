# **Diffusion-Model-LLM**

## **Overview**
This project integrates a **diffusion model** with **LLM-guided prompt refinement** to iteratively enhance text-to-image generation. The pipeline consists of **prompt extraction, image generation, and post-processing**. 

---

## **Installation & Setup**
### **1️⃣ Set Up the Environment**
Run the setup script to create a virtual environment and install dependencies:

```bash
bash scripts/setup.sh
```

---

### **2️⃣ Configure the Environment**
Ensure all necessary paths are set correctly:

```bash
source scripts/config.sh
```

This sets environment variables :
- `PYTHONPATH` → Source directory
- `PROMPTS_FILE` → Filtered prompts file
- `OUTPUT_DIR` → Output directory

---

### **3️⃣ Run the Full Pipeline**
Execute the entire workflow, from prompt extraction to image generation:

```bash
bash scripts/run.sh
```

This performs:
1. **Prompt Extraction:** Extracts relevant prompts from `evaluation_metadata.jsonl`.
2. **Diffusion Image Generation:** Generates images using the refined prompts.
3. **Post-Processing:** Organizes outputs into `generated_images/` for evaluation.

---

## **Project Structure**
```
project_root/
├── generated_images/                  # Final processed images
├── new_outputs/                       # Intermediate results
├── filtered_prompts/                  # Extracted prompts from metadata
├── evaluation_metadata.jsonl          # Input metadata file
├── scripts/                           # Scripts for running pipeline
│   ├── extract/                      
│   │   ├── extract_prompts.py         # Extracts prompts from JSONL file
│   ├── post_process/
│   │   ├── post_process.py            # Organizes outputs after image generation
│   ├── run.sh                         # Runs the full pipeline
│   ├── setup.sh                       # Sets up the environment
│   ├── clean.sh                       # Cleans up generated files
│   ├── config.sh                       # Exports paths & settings
│  
├── src/                               # Source code
│   ├── diffusion/                     # Diffusion model logic
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── refine.py
│   │   ├── config.py
│   │   ├── image_utils.py
│   │   └── models.py
│   ├── qwen_integration.py            # LLM integration for prompt refinement
│   └── main.py                        # Main pipeline script
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

---

## **Cleaning Up**
To remove generated outputs and reset the environment:

```bash
bash scripts/clean.sh
```

This will delete:

✅ The virtual environment (`venv/`)  
✅ All generated images (`generated_images/`)  
✅ Intermediate outputs (`new_outputs/`)  

---