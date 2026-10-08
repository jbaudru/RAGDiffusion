import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
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

def _disable_safety_checker(pipe):
    """Disable the diffusers safety checker to avoid returning black/blocked images.

    This replaces the pipeline's safety_checker with a dummy function that returns
    the original images and marks no NSFW content detected. Use with caution.
    """
    try:
        def dummy_safety(images, clip_input=None, **kwargs):
            return images, [False] * len(images)

        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = dummy_safety
        # Avoid calling the original run_safety_checker (which calls feature_extractor).
        # Replace it with a dummy that returns images unchanged and no NSFW flags.
        try:
            pipe.run_safety_checker = lambda images, device, dtype: (images, [False] * len(images))
        except Exception:
            # If assignment fails for some pipeline implementations, ignore and continue
            pass
    except Exception:
        # If anything goes wrong, don't crash — just continue with default behavior
        pass

def main(prompt, negative_prompt, video_path, output_name, fps, num_inference_steps, strength, guidance_scale, blend, num_previous_frames=1, seed=66, prompt_file=None, duration=None, width=512, height=512, model_id="stabilityai/sd-turbo"):
    video_converter = VideoConverter()
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        print(f"Using seed: {seed}")

    # Determine if we're using input video or generating from scratch
    use_input_video = video_path is not None and os.path.exists(video_path)
    if use_input_video:
        # Use img2img pipeline for video-to-video generation (Stable Diffusion)
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            _disable_safety_checker(pipe)
        except Exception:
            print(f"Failed to load Img2Img pipeline for model '{model_id}'. Make sure the model id is correct and you have access if it's gated.")
            raise
        print(f"Using input video: {video_path}")
    else:
        # Text-to-image/video generation
        try:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            _disable_safety_checker(pipe)
        except Exception:
            print(f"Failed to load Stable Diffusion pipeline for model '{model_id}'. Make sure the model id is correct and you have access if it's gated.")
            raise
        print(f"Generating video from text prompt only using model: {model_id}")

        # Calculate number of frames based on duration and fps
        if duration is None:
            duration = 5.0  # Default 5 seconds if not specified
        num_frames = int(duration * fps)
        print(f"Generating {num_frames} frames for {duration} seconds at {fps} fps")

        # move pipe to CUDA where applicable
        # FluxPipeline often supports model cpu offload and may run with bfloat16 on CPU; still move to cuda if possible
        try:
            pipe = pipe.to("cuda")
        except Exception:
            # If .to("cuda") fails (for example when using CPU offload), ignore and continue
            pass

        # try to disable progress bars where supported
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
    
    if use_input_video:
        temp_folder = "frames"
        os.makedirs(temp_folder, exist_ok=True)
        num_frames = video_converter.extract_frames(video_path, temp_folder, fps=fps)

    # Initialize PromptHandler if a prompt file is provided
    prompt_handler = None
    if prompt_file:
        prompt_handler = PromptHandler(prompt_file, fps=fps)

    previous_generated_images = []
    os.makedirs("temp_frames", exist_ok=True)
    
    for i in tqdm(range(num_frames)):
        # Calculate frame time
        frame_time = i / fps

        # Get the prompt for the current frame
        if prompt_handler:
            current_prompt = prompt_handler.get_prompt_for_frame(frame_time)
        else:
            current_prompt = prompt

        if use_input_video:
            # Image-to-image generation using input video frame
            original_image_path = f"frames/frame_{i:05d}.png"
            original_image = Image.open(original_image_path).convert("RGB")

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
        else:
            # Text-to-image generation for creating video from scratch
            # Stable Diffusion path (supports img2img blending continuity)
            if previous_generated_images and blend > 0:
                # For text-to-image, we can use img2img with the previous frame for continuity
                pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                pipe_img2img = pipe_img2img.to("cuda")
                pipe_img2img.set_progress_bar_config(disable=True)
                _disable_safety_checker(pipe_img2img)

                # Use the last generated frame as a base for continuity
                base_image = previous_generated_images[-1]

                # Blend with multiple previous frames if specified
                if len(previous_generated_images) > 1:
                    blended_image = base_image
                    for j, prev_image in enumerate(reversed(previous_generated_images[-num_previous_frames:])):
                        alpha = blend / (j + 1)
                        blended_image = Image.blend(blended_image, prev_image, alpha=alpha)
                else:
                    blended_image = base_image

                generated_image = pipe_img2img(
                    prompt=current_prompt,
                    negative_prompt=negative_prompt,
                    image=blended_image,
                    strength=0.7,  # Use moderate strength for frame continuity
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    disable_progress_bar=True,
                ).images[0]
            else:
                # Generate first frame or when blending is disabled
                generated_image = pipe(
                    prompt=current_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    disable_progress_bar=True,
                ).images[0]

        generated_image.save(f"temp_frames/frame_{i:05d}_generated.png")

        previous_generated_images.append(generated_image)
        if len(previous_generated_images) > num_previous_frames:
            previous_generated_images.pop(0)
    
    video_path = output_name + ".mp4"
    res_folder = "temp_frames"
    video_converter.frames_to_video(res_folder, video_path, fps=fps)

    video_converter.clean_temp_folders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Diffusion - Generate videos from prompts with or without input video")
    parser.add_argument("--prompt", type=str, default="painting pannel, saint face, french rococo style, embroidered cloak, garlands with ribbons and bows, garlands of mini roses, reliquary, laurels leafs and wild flowers, Francois Lemoyne design inspire, rosary, golden details, lace, halo, sky, rococo, baby torquoise, pink", help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="ugly, bad anatomy, blurry, pixelated, watermark, text, low quality, distorted", help="Negative prompt to specify what to avoid in generation")
    parser.add_argument("--video_path", type=str, help="Path to the input video file (optional - if not provided, generates video from text only)")
    parser.add_argument("--output_name", type=str, default="output_video/generated_video", help="Folder to save output video")
    parser.add_argument("--fps", type=int, default=16, help="Number of frames per second for the output video")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps for frame generation")
    parser.add_argument("--strength", type=float, default=0.5, help="Strength of the original video (0= Original, 1= Fully generated) - only used with input video")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for video generation (<7.5 = More creative freedom, >7.5 = More adherence to prompt)")
    parser.add_argument("--blend", type=float, default=0.05, help="Blending factor for frame blending (% of previous frame)")
    parser.add_argument("--num_previous_frames", type=int, default=1, help="Number of previous frames to blend")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--prompt_file", type=str, help="Path to a .txt or .srt file containing prompts")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of the generated video in seconds (only used when no input video)")
    parser.add_argument("--width", type=int, default=512, help="Width of generated frames (only used when no input video)")
    parser.add_argument("--height", type=int, default=512, help="Height of generated frames (only used when no input video)")
    parser.add_argument("--model_id", type=str, default="stabilityai/sd-turbo", help="Model id to use for generation (e.g. 'stabilityai/sd-turbo' or 'prompthero/openjourney')")

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
        prompt_file=args.prompt_file,
        duration=args.duration,
        width=args.width,
        height=args.height,
        model_id=args.model_id,
    )