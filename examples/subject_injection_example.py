"""
Multi-Subject Consistency Example for LongLive

This example demonstrates how to use the subject injection feature
to maintain consistent appearance of subjects across generated video frames.
"""

import torch
from PIL import Image
import torchvision.transforms as transforms

# Load your pipeline (assuming args is configured)
# from pipeline.causal_inference import CausalInferencePipeline
# pipeline = CausalInferencePipeline(args, device)


def load_reference_image(image_path: str, target_size: tuple = (480, 832)) -> torch.Tensor:
    """
    Load and preprocess a reference image.

    Args:
        image_path: Path to the reference image
        target_size: (height, width) to resize to

    Returns:
        tensor: [1, 3, H, W] in range [-1, 1]
    """
    img = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5], [0.5])  # [-1, 1]
    ])

    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def get_text_token_index(tokenizer, text: str, word: str) -> int:
    """
    Find the token index of a specific word in the text.

    Note: This is a simplified example. You may need to adjust based on
    your tokenizer's behavior (e.g., handling of subword tokenization).
    """
    tokens = tokenizer.tokenize(text)

    # Find the word (case-insensitive)
    word_lower = word.lower()
    for i, token in enumerate(tokens):
        if word_lower in token.lower():
            return i

    raise ValueError(f"Word '{word}' not found in text")


# Example usage
def example_inference():
    """
    Example of generating a video with subject consistency.
    """
    # Text prompt
    text_prompt = "A man in a red shirt walking with a golden retriever dog in a sunny park"

    # Reference images for subjects you want to keep consistent
    reference_subjects = {
        "man": {
            "image": load_reference_image("path/to/man_reference.jpg"),
            "token_idx": 2,  # Index of "man" in tokenized text (adjust based on your tokenizer)
            # Optional: if the word spans multiple tokens
            # "token_indices": [2, 3]
        },
        "dog": {
            "image": load_reference_image("path/to/dog_reference.jpg"),
            "token_idx": 10,  # Index of "dog" in tokenized text
        }
    }

    # Generate noise
    batch_size = 1
    num_frames = 120  # 40 blocks * 3 frames per block
    num_channels = 16
    height, width = 60, 104  # Latent space dimensions

    noise = torch.randn(
        batch_size, num_frames, num_channels, height, width,
        device='cuda', dtype=torch.bfloat16
    )

    # Run inference with subject injection
    # output = pipeline.inference(
    #     noise=noise,
    #     text_prompts=[text_prompt],
    #     reference_subjects=reference_subjects,
    #     injection_layers=[25, 26, 27, 28, 29],  # Last 5 transformer layers
    #     injection_strength=0.3,  # Adjust based on desired consistency vs creativity
    # )

    print("Reference subjects configured:")
    for name, info in reference_subjects.items():
        print(f"  - {name}: token_idx={info['token_idx']}, image_shape={info['image'].shape}")

    print("\nInjection parameters:")
    print("  - injection_layers: [25, 26, 27, 28, 29] (last 5 of 30 layers)")
    print("  - injection_strength: 0.3")
    print("\nReady for inference!")


if __name__ == "__main__":
    example_inference()
