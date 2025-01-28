import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class GenerationConfig:
    """Configuration parameters for image generation"""
    temperature: float = 1.0
    parallel_size: int = 16
    cfg_weight: float = 5.0
    image_token_num_per_image: int = 576
    img_size: int = 384
    patch_size: int = 16
    output_dir: str = 'generated_samples'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class JanusImageGenerator:
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize the Janus image generation model and processor

        Args:
            model_path: Path to the pretrained model
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Set inference mode globally for the model
        torch.set_grad_enabled(False)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        # Load and configure the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Move model to device and set precision
        if self.device == 'cuda':
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device).eval()

    def prepare_prompt(self, conversations: List[Dict[str, str]], system_prompt: str = "") -> str:
        """Prepare the conversation prompt for image generation"""
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversations,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=system_prompt,
        )
        return sft_format + self.vl_chat_processor.image_start_tag

    def prepare_input_tokens(self, prompt: str, config: GenerationConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tokens for generation"""
        input_ids = torch.LongTensor(self.tokenizer.encode(prompt))
        tokens = torch.zeros((config.parallel_size * 2, len(input_ids)), dtype=torch.int)
        tokens = tokens.to(self.device)

        for i in range(config.parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        return tokens, inputs_embeds

    @torch.inference_mode()
    def generate_tokens(self, inputs_embeds: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """Generate image tokens using the model"""
        # Ensure inputs are detached
        inputs_embeds = inputs_embeds.detach()
        generated_tokens = torch.zeros(
            (config.parallel_size, config.image_token_num_per_image),
            dtype=torch.int
        ).to(self.device)

        outputs = None
        for i in range(config.image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + config.cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / config.temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)],
                dim=1
            ).view(-1)

            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        return generated_tokens

    def decode_images(self, tokens: torch.Tensor, config: GenerationConfig) -> np.ndarray:
        """Decode image tokens into pixel arrays"""
        with torch.inference_mode():
            # Detach and clone the tokens to prevent autograd issues
            tokens = tokens.detach().clone()
            dec = self.model.gen_vision_model.decode_code(
                tokens.to(dtype=torch.int),
                shape=[config.parallel_size, 8,
                       config.img_size // config.patch_size,
                       config.img_size // config.patch_size]
            )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros(
            (config.parallel_size, config.img_size, config.img_size, 3),
            dtype=np.uint8
        )
        visual_img[:, :, :] = dec
        return visual_img

    def save_images(self, images: np.ndarray, config: GenerationConfig) -> List[str]:
        """Save generated images to disk"""
        os.makedirs(config.output_dir, exist_ok=True)
        saved_paths = []

        for i in range(config.parallel_size):
            save_path = os.path.join(config.output_dir, f"img_{i}.jpg")
            Image.fromarray(images[i]).save(save_path)
            saved_paths.append(save_path)

        return saved_paths

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> List[str]:
        """Generate images from a text prompt"""
        if config is None:
            config = GenerationConfig(device=self.device)
        else:
            config.device = self.device

        conversations = [
            {"role": "User", "content": prompt},
            {"role": "Assistant", "content": ""},
        ]

        formatted_prompt = self.prepare_prompt(conversations)
        tokens, embeddings = self.prepare_input_tokens(formatted_prompt, config)
        generated_tokens = self.generate_tokens(embeddings, config)
        images = self.decode_images(generated_tokens, config)
        return self.save_images(images, config)


def main():
    model_path = "deepseek-ai/Janus-Pro-1B"

    # Initialize generator with automatic device selection
    generator = JanusImageGenerator(model_path)

    prompt = """A close-up high-contrast photo of Sydney Opera House sitting next to 
    Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, 
    and radiating swirls of blue."""

    # For CPU, you might want to reduce parallel_size to save memory
    config = GenerationConfig(
        temperature=1.0,
        parallel_size=4 if torch.cuda.is_available() else 2,  # Reduced for CPU
        cfg_weight=5.0
    )

    image_paths = generator.generate(prompt, config)
    print(f"Generated images saved to: {image_paths}")


if __name__ == '__main__':
    main()