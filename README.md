# RAGDiffusion

This project implements a diffusion model for video-to-video generation, specifically for the task of generating videos from text prompts. The model is based on the diffusion model "SD Turbo", the principal idea is to use a previous generated frame of the video blended with the current frame to generate the next frame. This method allows for a more coherent and consistent video generation process, as the model can leverage the information from the previous frame to generate the next one.


## Installation
```bash
git clone https://github.com/jbaudru/RAGDiffusion.git 
cd RAGDiffusion
pip install -r requirements.txt
```


## Usage
```bash
python main.py --prompt "A cat playing with a ball" --video_path "test.mov" --output_name "resutl" --fps 10 --num_inference_steps 4 --strength 0.75 --guidance_scale 7.5 --blend 0.3
```

## Parameters
- `--prompt`: The text prompt to guide the image generation process (default: `"A futuristic cityscape at sunset"`).
- `--video_path`: Path to the input video file (default: `"input/video.mov"`).
- `--output_name`: Name of the output video (default: `"result"`).
- `--fps`: Frames per second for the output video (default: `10`).
- `--num_inference_steps`: Number of inference steps for image generation (default: `4`).
- `--strength`: Strength of the image blending (default: `0.75`).
- `--guidance_scale`: Guidance scale for image generation (default: `7.5`).
- `--blend`: Blending factor for mixing the current frame with the previous generated frame (default: `0.3`).

## Example

### Original Video
![Description](example/original.gif)
