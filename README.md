
We use a general template in `get_refined_prompt` (`src/qwen_integration.py` and `src/aoai.py`), selected per tag: one branch for `single_object`, one for `two_object`, and one shared branch for the remaining tags (`position`, `colors`, `counting`, `color_attr`, and a few internal variants). Each branch can be edited on its own, for example to add in-context examples specific to a category. We found TIR benefits from in-context examples.

## Benchmark results

**LLM-Grounded Diffusion benchmark [4], TIR improving SD-1.5, SD-2.1, Flux, and DALL-E 3 (MLLM: GPT-4o):**

| Method | Negation | Numeracy | Attribute | Spatial | Average |
|---|---|---|---|---|---|
| SD-1.5 | 35.0 | 40.0 | **42.0** | 38.0 | 38.75 |
| w/ TIR | **80.0** | **51.7** | 30.0 | **54.5** | **54.0** *(+15.25)* |
| SD-2.1 | 45.0 | 50.6 | 18.0 | **44.5** | 39.5 |
| w/ TIR | **85.0** | **56.0** | **18.0** | 43.0 | **50.5** *(+11.0)* |
| Flux | 10.0 | 52.2 | **83.0** | **81.5** | 56.7 |
| w/ TIR | **55.0** | **64.7** | 81.8 | 77.0 | **69.6** *(+12.9)* |
| DALL-E 3 | 31.6 | 44.5 | **73.0** | 81.0 | 57.5 |
| w/ TIR | **73.7** | **49.2** | 71.0 | **83.0** | **69.2** *(+11.7)* |

**GenEval [2], TIR improving Flux and DALL-E 3 (MLLM: GPT-4o):**

| Model | Position | Counting | Single Obj. | Two Object | Color Attr | Colors | Overall |
|---|---|---|---|---|---|---|---|
| Flux | 19.00 | 68.75 | **100.00** | 75.76 | **48.00** | 77.66 | 64.86 |
| w/ TIR | **49.00** | **71.25** | 98.75 | **80.81** | 47.00 | **80.85** | **71.27** *(+6.41)* |
| DALL-E 3 | 34.00 | 48.75 | **96.25** | 77.78 | 31.00 | 74.47 | 60.37 |
| w/ TIR | **45.00** | **60.00** | 96.25 | **82.83** | **38.00** | **86.17** | **68.04** *(+7.67)* |

**GenEval [2], TIR improving Flux (MLLM: Qwen2.5-VL-7B):**

| Model | Position | Counting | Single Obj. | Two Object | Color Attr | Colors | Overall |
|---|---|---|---|---|---|---|---|
| Flux | 19.00 | **68.75** | **100.00** | 75.76 | **48.00** | 77.66 | 64.86 |
| w/ TIR | **29.00** | 67.50 | 98.75 | **86.87** | 44.00 | **81.91** | **68.01** *(+3.15)* |

**DrawBench [3], TIR improving DALL-E 3 (MLLM: GPT-4o):**

<p align="center">
  <img src="assets/drawbench_results.png" width="500">
</p>

## References

[1] Khan, M. A. H., Jain, Y., Bhattacharyya, S., & Vineet, V. (2025). Test-time Prompt Refinement for Text-to-Image Models. *ICCV 2025 Workshops (MARS2)*, 6506-6516.

[2] Ghosh, D., Hajishirzi, H., & Schmidt, L. (2023). GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment. *NeurIPS 2023*.

[3] Saharia, C., et al. (2022). Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. *NeurIPS 2022*.

[4] Lian, L., Li, B., Yala, A., & Darrell, T. (2023). LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models. *arXiv:2305.13655*.

```bibtex
@inproceedings{khan2025test,
  title={Test-time prompt refinement for text-to-image models},
  author={Khan, Mohammed Abdul Hafeez and Jain, Yash and Bhattacharyya, Siddhartha and Vineet, Vibhav},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6565--6575},
  year={2025}
}
```

## Acknowledgement

This work builds on [Stable Diffusion](https://arxiv.org/abs/2112.10752), [SDXL](https://arxiv.org/abs/2307.01952), [Stable Diffusion 3](https://arxiv.org/abs/2403.03206), [SANA 1.5](https://arxiv.org/abs/2501.18427), [Flux](https://github.com/black-forest-labs/flux), [DALL-E 3](https://cdn.openai.com/papers/dall-e-3.pdf), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), and GPT-4o (OpenAI). We thank their authors for making these models available and sharing their work with the community.