import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
import argparse
import numpy as np

def modify_image(prompt, negative_prompt, input_image_path, output_image_path, num_inference_steps, strength, guidance_scale, seed=66, model_id=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Using seed: {seed}")

    # Use a safe default model unless the caller provides one.
    if model_id is None:
        # Default to a public, generally-available model to avoid gated-repo errors
        model_id = "stabilityai/sd-turbo"
        model_id = "stabilityai/sd-2-1-base"
        model_id = "runwayml/stable-diffusion-v1-5"
        

    try:
        if input_image_path is not None:
            # Use Img2Img pipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")  # Use GPU for faster inference
        else:
            # Use Txt2Img pipeline
            # Special-case Flux-like ids: user can pass a Flux model id explicitly; we import FluxPipeline only when needed.
            if "flux" in model_id.lower():
                try:
                    from diffusers import FluxPipeline
                except Exception:
                    # If FluxPipeline isn't available in this diffusers build, fall back to raising original import error
                    raise

                pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
                # try enabling CPU offload where available
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass
            else:
                pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                pipe = pipe.to("cuda")  # Use GPU for faster inference
                pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        # Provide a clearer message for gated-repo errors (common with some FLUX models)
        err_str = str(e)
        print(f"Error loading model '{model_id}': {err_str}")
        print("This may be because the model is gated or requires authentication on Hugging Face.")
        print("To fix: 1) Run 'huggingface-cli login' and provide a valid token, or 2) set the HUGGINGFACE_HUB_TOKEN environment variable.")
        print("Example (PowerShell):")
        print("  $env:HUGGINGFACE_HUB_TOKEN = 'hf_xxx' ; python .\\genImg.py --model_id 'stabilityai/sd-turbo'")
        print("Or use a public model id via --model_id (e.g. 'stabilityai/sd-turbo' or 'runwayml/stable-diffusion-v1-5').")
        raise

    # Generation stage: produce image(s) after the pipeline 'pipe' has been created
    if input_image_path is not None:
        # If user provided an input image, we run Img2Img on Stable Diffusion pipelines only
        if "flux" in model_id.lower():
            print("Flux models are not supported for Img2Img in this script. Please provide a non-Flux model for --model_id or omit --input_image_path.")
            raise SystemExit(1)

        # Load the input image
        input_image = Image.open(input_image_path).convert("RGB")

        # Generate the modified image
        generated_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            disable_progress_bar=True,
        ).images[0]
    else:
        # Text-only generation
        print("No input image provided. Using Txt2Img pipeline.")

        # Generate the image from text
        if "flux" in model_id.lower():
            # FluxPipeline may return images in result.images
            result = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=1080,
                width=1920,
            )
            generated_image = result.images[0]
        else:
            generated_image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=1080,
                width=1920,
                disable_progress_bar=True,
            ).images[0]

    # Save the generated image
    generated_image.save(output_image_path)
    print(f"Generated image saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Modifier using Stable Diffusion")
    parser.add_argument("--prompt", type=str, default="renaissance painting, Black and red demon", help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="ugly, bad anatomy, blurry, pixelated, watermark, text, low quality, distorted", help="Negative prompt to specify what to avoid in generation")
    parser.add_argument("--input_image_path", type=str, help="Path to the input image file")
    parser.add_argument("--output_image_path", type=str, default="output_img/rock0.png", help="Path to save the output image")
    parser.add_argument("--num_inference_steps", type=int, default=5, help="Number of inference steps for image generation")
    parser.add_argument("--strength", type=float, default=0.45, help="Strength of the original image (only used for Img2Img)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for image generation (<7.5 = More creative freedom, >7.5 = More adherence to prompt)")
    parser.add_argument("--seed", type=int, default=66, help="Random seed for reproducibility")
    parser.add_argument("--model_id", type=str, default="stabilityai/sd-turbo", help="Model id to load (use a public model or ensure you are authenticated for gated models)")

    # model_id = "prompthero/openjourney"
    # model_id = "black-forest-labs/FLUX.1-Krea-dev"
    # model_id = "black-forest-labs/FLUX.1-dev"
    # model_id = "black-forest-labs/FLUX.1-schnell"
    

    args = parser.parse_args()

    modify_image(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image_path=args.input_image_path,
        output_image_path=args.output_image_path,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        model_id=args.model_id,
    )