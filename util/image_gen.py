import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import argparse
import numpy as np

def modify_image(prompt, negative_prompt, input_image_path, output_image_path, num_inference_steps, strength, guidance_scale, seed=66):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Using seed: {seed}")

    #model_id = "stabilityai/sd-turbo"  # Smaller and faster than v1-5
    model_id = "CompVis/stable-diffusion-v1-4"  # Pruned model for faster inference
    model_id = "runwayml/stable-diffusion-v1-5"  # Original model for better quality
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU for faster inference

    if input_image_path is not None:
        # Load the input image
        input_image = Image.open(input_image_path).convert("RGB")
    else:
        # Create a blank image with default resolution (1080x1920)
        noise = np.random.rand(1080, 1920, 3) * 255  # Random noise in the range [0, 255]
        input_image = Image.fromarray(noise.astype('uint8')).convert("RGB")
        strength = 0.95
        print("No input image provided. Using a blank image with resolution 1080x1920.")

    # Generate the modified image
    generated_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        disable_progress_bar=True
    ).images[0]

    # Save the generated image
    generated_image.save(output_image_path)
    print(f"Generated image saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Modifier using Stable Diffusion")
    parser.add_argument("--prompt", type=str, default="painting of castle in the sky", help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="ugly, bad anatomy, blurry, pixelated, watermark, text, low quality, distorted", help="Negative prompt to specify what to avoid in generation")
    parser.add_argument("--input_image_path", type=str, help="Path to the input image file")
    parser.add_argument("--output_image_path", type=str, default="img_out/out_v1_5.png", help="Path to save the output image")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for image generation")
    parser.add_argument("--strength", type=float, default=0.45, help="Strength of the original image (0=Original, 1=Fully generated)")
    parser.add_argument("--guidance_scale", type=float, default=8.5, help="Guidance scale for image generation (<7.5 = More creative freedom, >7.5 = More adherence to prompt)")
    parser.add_argument("--seed", type=int, default=66, help="Random seed for reproducibility")

    args = parser.parse_args()

    modify_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image_path=args.input_image_path,
        output_image_path=args.output_image_path,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )