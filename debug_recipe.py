import sys
import torch
from omegaconf import OmegaConf
from torchtune.config import instantiate
from recipes.lora_finetune_single_device import LoRAFinetuneRecipeSingleDevice

def debug_recipe_setup(config_path):
    print(f"\n=== Testing Recipe Setup: {config_path} ===")
    
    # Load config exactly like the recipe does
    cfg = OmegaConf.load(config_path)
    
    # Create recipe instance
    recipe = LoRAFinetuneRecipeSingleDevice(cfg=cfg)
    
    # Try the setup that's failing
    try:
        print("Starting recipe setup...")
        recipe.setup(cfg=cfg)
        print("✓ Recipe setup successful!")
        return True
    except Exception as e:
        print(f"✗ Recipe setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    config_7b = "custom-configs/7B_lora_single_device.yaml"
    config_8b = "custom-configs/8B_lora_single_device.yaml"
    
    print("Testing 7B recipe setup...")
    debug_recipe_setup(config_7b)
    
    print("\nTesting 8B recipe setup...")
    debug_recipe_setup(config_8b)