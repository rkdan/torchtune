import sys
from omegaconf import OmegaConf
from torchtune.config import instantiate

def debug_full_instantiation(config_path):
    print(f"\n=== Testing {config_path} ===")
    
    # Load config
    cfg = OmegaConf.load(config_path)
    print(f"Model config: {cfg.model._component_}")
    print(f"Loss config: {cfg.loss}")
    
    # Test each component step by step
    components_to_test = [
        ("tokenizer", cfg.tokenizer),
        ("model", cfg.model),
        ("checkpointer", cfg.checkpointer),
        ("dataset", cfg.dataset),
        ("optimizer", cfg.optimizer),
        ("loss", cfg.loss),
    ]
    
    for name, component_cfg in components_to_test:
        try:
            print(f"\n--- Testing {name} instantiation ---")
            print(f"Component: {component_cfg._component_}")
            
            if name == "loss":
                # Test loss import directly before instantiation
                try:
                    from torchtune.modules.loss import LinearCrossEntropyLoss
                    print("✓ Direct loss import successful")
                except Exception as e:
                    print(f"✗ Direct loss import failed: {e}")
            
            component = instantiate(component_cfg)
            print(f"✓ {name} instantiation successful: {type(component)}")
            
        except Exception as e:
            print(f"✗ {name} instantiation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n✓ All components instantiated successfully!")
    return True

def debug_loss_only(config_path):
    print(f"\n=== Testing LOSS ONLY from {config_path} ===")
    
    # Load config
    cfg = OmegaConf.load(config_path)
    print(f"Loss config: {cfg.loss}")
    
    # Try to instantiate
    try:
        print("About to instantiate...")
        loss_fn = instantiate(cfg.loss)
        print(f"Success! Loss function: {loss_fn}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test both configs
    config_7b = "custom-configs/7B_lora_single_device.yaml"
    config_8b = "custom-configs/8B_lora_single_device.yaml"
    
    print("="*50)
    print("TESTING LOSS ONLY")
    print("="*50)
    
    print("Testing 7B config loss...")
    debug_loss_only(config_7b)
    
    print("\nTesting 8B config loss...")
    debug_loss_only(config_8b)
    
    print("\n" + "="*50)
    print("TESTING FULL INSTANTIATION")
    print("="*50)
    
    print("Testing 7B config full...")
    debug_full_instantiation(config_7b)
    
    print("\nTesting 8B config full...")
    debug_full_instantiation(config_8b)