<p align="left">
  <img src="https://img.shields.io/badge/Torch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/StableDiffusion-000000?style=for-the-badge&logo=stable%20diffusion&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

# RAGDiffusion

This project implements a diffusion model for **video-to-video** generation, specifically for the task of generating videos from text prompts. The model is based on the diffusion model "SD Turbo", **the principal idea is to use a previous generated frame of the video blended with the current frame to generate the next frame**. This method allows for a more coherent and consistent video generation process, as the model can leverage the information from the previous frame to generate the next one.


## Installation
```bash
git clone https://github.com/jbaudru/RAGDiffusion.git 
cd RAGDiffusion
pip install -r requirements.txt
```


## Usage
```bash
python video_gen.py --prompt "A cat playing with a ball" --video_path "test.mov" --output_name "resutl" --fps 10 --num_inference_steps 4 --strength 0.75 --guidance_scale 7.5 --blend 0.3
```

## Parameters
- `--prompt`: The text prompt to guide the image generation process (default: `"painting, trippy, psychadelic, sky, cloud, rainbow, 60's, 4k"`).
- `--negative_prompt`: Negative prompt to specify what to avoid in generation (default: `"ugly, bad anatomy, blurry, pixelated, watermark, text, low quality, distorted"`).
- `--video_path`: Path to the input video file (default: `"input/dance.mp4"`).
- `--output_name`: Name of the output video without extension (default: `"output_video/test_srtfile"`).
- `--fps`: Frames per second for the output video (default: `5`).
- `--num_inference_steps`: Number of inference steps for frame generation (default: `5`).
- `--strength`: Strength of the diffusion effect (0 = Original video, 1 = Fully generated) (default: `0.4`).
- `--guidance_scale`: Guidance scale for video generation - lower values (<7.5) provide more creative freedom, higher values (>7.5) enforce stricter adherence to the prompt (default: `7.5`).
- `--blend`: Blending factor for mixing the current frame with the previous generated frame (default: `0.4`).
- `--num_previous_frames`: Number of previous frames to blend (default: `2`).
- `--seed`: Random seed for reproducibility (default: `66`).
- `--prompt_file`: Path to a `.txt` or `.srt` file containing prompts. 

You can easily generate an `.srt` file for a video using this tool: [Clideo SRT File Generator](https://clideo.com/create-srt-file).


## Runtime
For **stabilityai/sd-turbo** model:
| GPU |  num_inference_steps | strength | guidance_scale | blend | runtime(s)/frame |
|-----|----------------------|----------|----------------|-------|-------------|
| RTX 3090 | 10 | 0.25 | 7.5 | 0.15 | 2.16s |
| RTX 3090 | 15 | 0.25 | 7.5 | 0.15 | 2.57s |
| RTX 3090 | 15 | 0.35 | 7.5 | 0.15 | 3.53s |
| RTX 3090 | 25 | 0.25 | 7.5 | 0.15 | 3.88s |

*Todo: Test runtime when using .srt file, using multiple prompts, and using a different model.*


## Example

### Original Video
![Description](example/original.gif)

### 0% Blend (No RAG)
![Description](example/rag0.gif)
<sub>**Parameters**: fps:24, num_inference_steps:10, strength:0.35, guidance_scale:7.5, blend:0</sub>

### 5% Blend (RAG)
![Description](example/rag05.gif)
<sub>**Parameters**: fps:24, num_inference_steps:10, strength:0.35, guidance_scale:7.5, blend:0.5</sub>

### 15% Blend (RAG)
![Description](example/rag15.gif)
<sub>**Parameters**: fps:24, num_inference_steps:10, strength:0.35, guidance_scale:7.5, blend:0.15</sub>

### 35% Blend (RAG)
![Description](example/rag35.gif)
<sub>**Parameters**: fps:24, num_inference_steps:10, strength:0.35, guidance_scale:7.5, blend:0.35</sub>


*Todo: Show difference with 1 or multiple previous frames*

## TODO
- Keep sound of original video and add it to the generated video.
- Speed up the generation process.
