import os
import numpy as np
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import cv2
from IPython.display import display, Image as IPImage

def inference():
    # Prompt user to upload an image
    print("Please upload an image to process:")
    uploaded = files.upload()  # Upload functionality in Colab

    if not uploaded:
        print("No file uploaded. Exiting.")
        return

    # Directory to save the uploaded image(s)
    input_save_folder = "input_images"
    os.makedirs(input_save_folder, exist_ok=True)

    # Save each uploaded image file to the designated folder
    for filename in uploaded.keys():
        original_image_path = os.path.join(input_save_folder, filename)
        with open(original_image_path, 'wb') as f:
            f.write(uploaded[filename])

        # Load the saved image
        image = np.array(Image.open(original_image_path))
        display(IPImage(filename=original_image_path))  # Display the uploaded image

        # Resize for faster processing
        image = np.array(Image.open(original_image_path).resize((256, 256)))

        # Apply Canny Edge Detection
        edges = cv2.Canny(image, 100, 200)
        edges = np.stack([edges] * 3, axis=-1)
        canny_image = Image.fromarray(edges)

        # Display the Canny edge image
        print("Canny Edge Detected Image:")
        display(canny_image)

        # Load the fine-tuned model
        model_id = "my-ctrlnet"  # Replace with your fine-tuned model path
        pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipeline = pipeline.to("cuda")

        # Get user input for the design prompt
        prompt = input("Enter your design prompt: ")

        # Generate the output image
        try:
            print(f"Generating image with prompt: '{prompt}'...")
            output = pipeline(
                prompt=prompt,
                image=canny_image,
                negative_prompt="low quality, blurry, distorted shapes",
                num_inference_steps=20,
            ).images[0]
        except Exception as e:
            print(f"Error during image generation: {e}")
            return

        # Display and save the output image
        print("Generated Image:")
        display(output)

        # Save the output image
        save_folder = "output_images"
        os.makedirs(save_folder, exist_ok=True)
        output_image_path = os.path.join(save_folder, f"generated_output_{filename}")
        output.save(output_image_path)
        print(f"Output image saved to {output_image_path}")

if __name__ == "__main__":
    inference()
