import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from util.converter import VideoConverter
import argparse
import os
import shutil
from tqdm import tqdm
import random
import numpy as np

def main(prompt, video_path, output_name, fps, num_inference_steps, strength, guidance_scale, blend, seed=66):
    video_converter = VideoConverter()
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Using seed: {seed}")

    model_id = "stabilityai/sd-turbo"  # Smaller and faster than v1-5
    # Alternative options:
    # - "runwayml/stable-diffusion-v1-5-pruned" (pruned model)
    # - "stabilityai/sd-turbo" (optimized for speed)
    # - "CompVis/stable-diffusion-v1-4"
    # - "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU for faster inference
    #pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    temp_folder = "frames"
    # Create output folder if it doesn't exist
    os.makedirs(temp_folder, exist_ok=True)
    num_frames = video_converter.extract_frames(video_path, temp_folder, fps=fps)

    # Keep track of previous generated frame
    last_generated_image = None

    for i in tqdm(range(num_frames)):
        # Load the frame image  
        original_image_path = f"frames/frame_{i:05d}.png"
        original_image = Image.open(original_image_path).convert("RGB")
        
        # For frames after the first one, blend with previous generated image
        if last_generated_image is not None:
            # Blend current frame with previous generated frame 
            blended_image = Image.blend(original_image, last_generated_image, alpha=blend)
        else:
            blended_image = original_image

        #blended_image = video_converter.resize_for_processing(blended_image)
        
        generated_image = pipe(
            prompt=prompt, 
            image=blended_image, 
            strength=strength, 
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            disable_progress_bar=True
        ).images[0]

        # Save and display the result
        os.makedirs("output_frames", exist_ok=True)
        generated_image.save(f"output_frames/frame_{i:05d}_generated.png")
        #generated_image.show()
        
        last_generated_image = generated_image
    

    video_path = output_name + ".mp4"
    res_folder = "output_frames"
    video_converter.frames_to_video(res_folder, video_path, fps=fps)

    # Clean up temporary files
    clean_temp_folders()

def clean_temp_folders():
    """Remove temporary frame folders after processing"""
    folders_to_clean = ["frames", "output_frames"]
    
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"Successfully deleted {folder} folder")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Diffusion")
    parser.add_argument("--prompt", type=str, default="Dali painting, surrealism, abstract", help="Text prompt for image generation")
    parser.add_argument("--video_path", type=str, default="input/dance.mp4", help="Path to the input video file")
    parser.add_argument("--output_name", type=str, default="output/dance15", help="Folder to save output video")
    
    parser.add_argument("--fps", type=int, default=24, help="Robotic, dangerous, and scary")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Number of inference steps for frame generation")
    parser.add_argument("--strength", type=float, default=0.35, help="Strength of the original video (0= Orginal, 1= Fully generated)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for video generation (<7.5 = More creative freedom, >7.5 = More adherence to prompt)")
    parser.add_argument("--blend", type=float, default=0.15, help="Blending factor for frame blending (% of previous frame)")
    
    main(
        prompt=parser.parse_args().prompt, 
        video_path=parser.parse_args().video_path, 
        output_name=parser.parse_args().output_name, 
        fps=parser.parse_args().fps, 
        num_inference_steps=parser.parse_args().num_inference_steps, 
        strength=parser.parse_args().strength, 
        guidance_scale=parser.parse_args().guidance_scale,
        blend=parser.parse_args().blend
    )
    