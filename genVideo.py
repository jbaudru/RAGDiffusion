import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from util.converter import VideoConverter
from util.prompt_handler import PromptHandler 
import argparse
import os
import shutil
from tqdm import tqdm
import random
import numpy as np
import logging

def main(prompt, negative_prompt, video_path, output_name, fps, num_inference_steps, strength, guidance_scale, blend, num_previous_frames=1, seed=66, prompt_file=None):
    video_converter = VideoConverter()
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        print(f"Using seed: {seed}")

    model_id = "stabilityai/sd-turbo"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    
    temp_folder = "frames"
    os.makedirs(temp_folder, exist_ok=True)
    num_frames = video_converter.extract_frames(video_path, temp_folder, fps=fps)

    # Initialize PromptHandler if a prompt file is provided
    prompt_handler = None
    if prompt_file:
        prompt_handler = PromptHandler(prompt_file, fps=fps)

    previous_generated_images = []
    for i in tqdm(range(num_frames)):
        original_image_path = f"frames/frame_{i:05d}.png"
        original_image = Image.open(original_image_path).convert("RGB")
        
        # Calculate frame time
        frame_time = i / fps

        # Get the prompt for the current frame
        if prompt_handler:
            current_prompt = prompt_handler.get_prompt_for_frame(frame_time)
        else:
            current_prompt = prompt

        # Blend with previous frames
        if previous_generated_images:
            blended_image = original_image
            for j, prev_image in enumerate(reversed(previous_generated_images[-num_previous_frames:])):
                alpha = blend / (j + 1)
                blended_image = Image.blend(blended_image, prev_image, alpha=alpha)
        else:
            blended_image = original_image

        generated_image = pipe(
            prompt=current_prompt, 
            negative_prompt=negative_prompt,
            image=blended_image, 
            strength=strength, 
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            disable_progress_bar=True,
        ).images[0]

        os.makedirs("temp_frames", exist_ok=True)
        generated_image.save(f"temp_frames/frame_{i:05d}_generated.png")
        
        previous_generated_images.append(generated_image)
        if len(previous_generated_images) > num_previous_frames:
            previous_generated_images.pop(0)
    
    video_path = output_name + ".mp4"
    res_folder = "temp_frames"
    video_converter.frames_to_video(res_folder, video_path, fps=fps)

    video_converter.clean_temp_folders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Diffusion")
    parser.add_argument("--prompt", type=str, default="painting, trippy, psychadelic, sky, cloud, rainbow, 60's, 4k", help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="ugly, bad anatomy, blurry, pixelated, watermark, text, low quality, distorted", help="Negative prompt to specify what to avoid in generation")
    parser.add_argument("--video_path", type=str, default="input/dance.mp4", help="Path to the input video file")
    parser.add_argument("--output_name", type=str, default="output_video/dance_srtprompt_full", help="Folder to save output video")
    parser.add_argument("--fps", type=int, default=25, help="Number of frames per second for the output video")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps for frame generation")
    parser.add_argument("--strength", type=float, default=0.3, help="Strength of the original video (0= Orginal, 1= Fully generated)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for video generation (<7.5 = More creative freedom, >7.5 = More adherence to prompt)")
    parser.add_argument("--blend", type=float, default=0.3, help="Blending factor for frame blending (% of previous frame)")
    parser.add_argument("--num_previous_frames", type=int, default=6, help="Number of previous frames to blend")
    parser.add_argument("--seed", type=int, default=66, help="Random seed for reproducibility")
    parser.add_argument("--prompt_file", type=str, help="Path to a .txt or .srt file containing prompts")

    args = parser.parse_args()
    main(
        prompt=args.prompt, 
        negative_prompt=args.negative_prompt,
        video_path=args.video_path, 
        output_name=args.output_name, 
        fps=args.fps, 
        num_inference_steps=args.num_inference_steps, 
        strength=args.strength, 
        guidance_scale=args.guidance_scale,
        blend=args.blend,
        num_previous_frames=args.num_previous_frames,
        seed=args.seed,
        prompt_file=args.prompt_file
    )