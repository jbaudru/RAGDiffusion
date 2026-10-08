#!/usr/bin/env python3
"""
Example script demonstrating how to use the improved genVideo.py 
to generate videos from text prompts without requiring input video.
"""

import subprocess
import os

def run_text_to_video():
    """
    Generate a video from a text prompt only (no input video required)
    """
    print("=== Generating video from text prompt only ===")
    
    cmd = [
        "python", "genVideo.py",
        "--prompt", "dali painting style, a white and black cat, sleeping on a blanket, and cloud moving in the window",
        "--negative_prompt", "ugly, low quality, blurry, distorted, watermark",
        "--duration", "4.0",  # 8 seconds
        "--fps", "24",
        "--width", "1080",
        "--height", "512",
        "--output_name", "output_video/test2",
        "--num_inference_steps", "30",
        "--guidance_scale", "7.0",
        "--blend", "0.9",
        "--seed", "42"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Video generated successfully: output_video/magical_forest.mp4")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating video: {e}")

def run_with_prompt_file():
    """
    Generate a video using a prompt file with time-based prompts
    """
    print("\n=== Generating video with prompt file ===")
    
    # Create a sample prompt file
    prompt_content = """0:00 - A peaceful sunrise over mountains, golden hour lighting
0:03 - Birds flying across the sky, morning mist
0:06 - A flowing river through the valley, serene landscape"""
    
    with open("sample_prompts.txt", "w") as f:
        f.write(prompt_content)
    
    cmd = [
        "python", "genVideo.py",
        "--prompt_file", "sample_prompts.txt",
        "--duration", "10.0",
        "--fps", "10",
        "--output_name", "output_video/landscape_sequence",
        "--num_inference_steps", "12",
        "--blend", "0.2"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Video with prompt file generated successfully: output_video/landscape_sequence.mp4")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating video with prompt file: {e}")

def run_with_input_video():
    """
    Generate a video using both input video and prompt (traditional mode)
    """
    print("\n=== Generating video with input video (traditional mode) ===")
    
    # Check if input video exists
    input_video = "input/fumseck_small_short.mp4"
    if not os.path.exists(input_video):
        print(f"⚠️  Input video not found: {input_video}")
        print("Skipping traditional video-to-video generation")
        return
    
    cmd = [
        "python", "genVideo.py",
        "--prompt", "cyberpunk style, neon lights, futuristic cityscape",
        "--video_path", input_video,
        "--output_name", "output_video/cyberpunk_style",
        "--strength", "0.4",
        "--blend", "0.1"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Video with input generated successfully: output_video/cyberpunk_style.mp4")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating video with input: {e}")

if __name__ == "__main__":
    print("RAG Diffusion - Text-to-Video Examples")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs("output_video", exist_ok=True)
    
    # Run examples
    run_text_to_video()
    #run_with_prompt_file()
    #run_with_input_video()
    
    #print("\n" + "=" * 50)
    #print("All examples completed! Check the output_video/ folder for results.")
