import json
import random
from typing import Dict, List, Optional

class PromptAugmenter:
    """
    Prompt augmentation for LIBERO-Goal LoRA finetuning.
    Loads a mapping from original prompts to conversational variants.
    With a set probability, swaps the prompt for a conversational variant.
    """
    def __init__(self, variants_json_path: str, augmentation_prob: float = 0.5):
        self.augmentation_prob = augmentation_prob
        self.prompt_map = self._load_variants(variants_json_path)

    def _load_variants(self, path: str) -> Dict[str, List[str]]:
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def augment(self, prompt: str) -> str:
        """
        With probability augmentation_prob, return a random conversational variant.
        Otherwise, return the original prompt.
        """
        variants = self.prompt_map.get(prompt)
        if variants and random.random() < self.augmentation_prob:
            return random.choice(variants)
        return prompt

    def set_prob(self, prob: float):
        self.augmentation_prob = prob

    def get_variants(self, prompt: str) -> Optional[List[str]]:
        return self.prompt_map.get(prompt)
