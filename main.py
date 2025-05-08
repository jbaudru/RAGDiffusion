import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from util.converter import VideoConverter
import argparse
import os
import shutil

def main(prompt, video_path, output_folder, fps, num_inference_steps, strength, guidance_scale, blend):
    video_converter = VideoConverter()

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

    for i in range(num_frames):
        # Load the frame image  
        original_image_path = f"frames/frame_{i:05d}.png"
        original_image = Image.open(original_image_path).convert("RGB")
        
        # For frames after the first one, blend with previous generated image
        if last_generated_image is not None:
            # Blend current frame with previous generated frame (70% original, 30% previous)
            blended_image = Image.blend(original_image, last_generated_image, alpha=blend)
        else:
            blended_image = original_image

        #blended_image = video_converter.resize_for_processing(blended_image)
        
        generated_image = pipe(
            prompt=prompt, 
            image=blended_image, 
            strength=strength, 
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps  # Add this parameter
        ).images[0]

        # Save and display the result
        os.makedirs("output_frames", exist_ok=True)
        generated_image.save(f"output_frames/frame_{i:05d}_generated.png")
        #generated_image.show()
        
        last_generated_image = generated_image
    
    os.makedirs("output", exist_ok=True)
    video_path = output_folder + "result.mp4"
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
    parser.add_argument("--video_path", type=str, default="input/video.mov", help="Path to the input video file")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to save output video")
    
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video")
    parser.add_argument("--num_inference_steps", type=int, default=2, help="Number of inference steps for frame generation")
    parser.add_argument("--strength", type=float, default=0.7, help="Strength of the original video")
    parser.add_argument("--guidance_scale", type=float, default=5.5, help="Guidance scale for video generation")
    parser.add_argument("--blend", type=float, default=0.2, help="Blending factor for frame blending")
    
    main(
        prompt=parser.parse_args().prompt, 
        video_path=parser.parse_args().video_path, 
        output_folder=parser.parse_args().output_folder, 
        fps=parser.parse_args().fps, 
        num_inference_steps=parser.parse_args().num_inference_steps, 
        strength=parser.parse_args().strength, 
        guidance_scale=parser.parse_args().guidance_scale,
        blend=parser.parse_args().blend
    )
    