# Text-to-Image Generator Using Stable Diffusion

## Overview
This project is a Python-based **text-to-image generator** that uses Stable Diffusion to transform text prompts into high-quality images. It leverages the `diffusers` and `transformers` libraries from Hugging Face to generate AI-powered images based on user input.

## Features
- Converts textual descriptions into high-quality AI-generated images.
- Uses **Stable Diffusion 2** for image generation.
- Customizable settings such as inference steps, guidance scale, and image resolution.
- Works on both CPU and GPU.
- Saves generated images automatically.

## Installation
To use this project, ensure you have Python installed, then run the following commands:

```bash
pip install --upgrade diffusers transformers torch tqdm numpy pandas opencv-python matplotlib
```

## Usage
### 1. Load the Model
```python
from diffusers import StableDiffusionPipeline
import torch
import os

# Load Hugging Face Token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Initialize Stable Diffusion model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    revision='fp16',
    use_auth_token=hf_token
).to("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. Generate an Image
```python
def generate_image(prompt, model):
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Invalid prompt: Provide a non-empty string.")
    
    image = model(prompt, num_inference_steps=20, guidance_scale=9).images[0]
    image.thumbnail((400, 400))
    save_path = f"generated_images/{prompt.replace(' ', '_')}.png"
    image.save(save_path)
    print(f"Image saved at: {save_path}")
    return image

# Example usage
generate_image("A Horse", image_gen_model)
```

## Demo & Resources
- Live Demo: [https://colab.research.google.com/drive/1SFBNPWpIwZLrK2VtVqUE9aBRwQWs4cR8?usp=sharing]
- Presentation Link : [https://dituni-my.sharepoint.com/:p:/g/personal/1000017330_dit_edu_in/EYcDnEwp4WlKkhP7abyYKZ0BnmCdhkjhADCyUpfTKSbUDg?e=XZkoK6]
- for the best and fast result use kernel(Hardware Connector - T4 GPU

## Future Enhancements
- Improve prompt generation using larger language models.
- Increase image resolution and quality.
- Develop a web-based interface for user-friendly interaction.

## Author
Shashank Kanaujia 🚀

