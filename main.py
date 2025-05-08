import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from RAGDiffusion.util.convertererter import VideoConverter
import argparse
import os

def main(prompt, video_path, output_folder, fps, num_inference_steps, strength, guidance_scale, blend):
    video_converter = VideoConverter()

    # Load the pretrained Stable Diffusion model
    #model_id = "runwayml/stable-diffusion-v1-5"

    model_id = "stabilityai/sd-turbo"  # Smaller and faster than v1-5
    # Alternative lightweight options:
    # - "runwayml/stable-diffusion-v1-5-pruned" (pruned model)
    # - "stabilityai/sd-turbo" (optimized for speed)
    # - "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU for faster inference
    #pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()

    # Define the text prompt
    #prompt =  "A futuristic cityscape at sunset"


    # Extract frames from a video file (optional)
    #video_path = "input/video.mov"
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

        # Load the original image
        #original_image_path = "input/test.png"
        #original_image = Image.open(original_image_path).convert("RGB")
        
        # For frames after the first one, blend with previous generated image
        if last_generated_image is not None:
            # Blend current frame with previous generated frame (70% original, 30% previous)
            blended_image = Image.blend(original_image, last_generated_image, alpha=blend)
        else:
            blended_image = original_image


        #blended_image = VideoConverter.resize_for_processing(blended_image)
        
        # Generate the new image
        #strength = 0.75 # Higher strength means more influence from the original image
        #guidance_scale = 7.5 # Higher guidance scale means more adherence to the prompt
        #num_inference_steps = 4  # Reduce from default ~25-50 steps to just 8 steps
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Diffusion")
    parser.add_argument("--prompt", type=str, default="A futuristic cityscape at sunset", help="Text prompt for image generation")
    parser.add_argument("--video_path", type=str, default="input/video.mov", help="Path to the input video file")
    parser.add_argument("--output_folder", type=str, default="output", help="Folder to save output video")
    
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for output video")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps for image generation")
    parser.add_argument("--strength", type=float, default=0.75, help="Strength of the image blending")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for image generation")
    parser.add_argument("--blend", type=float, default=0.3, help="Blending factor for image blending")
    
    main()
    
    """
    video_converter = VideoConverter()  # Create an instance first
    video_path = "output/result_no_rag.mp4"
    output_folder = "output_frames"
    video_converter.frames_to_video(output_folder, video_path, fps=10)
    """