import torch
from model.Text_IF_model import Text_IF
import clip


def check_weights_compatibility(model, checkpoint_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model state dict - it's nested under 'model' key
    checkpoint_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Get current model state dict, excluding CLIP weights
    model_state_dict = {k: v for k, v in model.state_dict().items() 
                       if not k.startswith('model_clip.')}
    
    # Compare shapes
    mismatched_layers = []
    missing_keys = []
    extra_keys = []
    
    # Check for shape mismatches
    for key in model_state_dict.keys():
        if key not in checkpoint_state_dict:
            missing_keys.append(key)
            continue
            
        if checkpoint_state_dict[key].shape != model_state_dict[key].shape:
            mismatched_layers.append((
                key,
                checkpoint_state_dict[key].shape,
                model_state_dict[key].shape
            ))
    
    # Check for extra keys in checkpoint
    for key in checkpoint_state_dict.keys():
        if key not in model_state_dict:
            extra_keys.append(key)
    
    # Print results
    if len(mismatched_layers) == 0 and len(missing_keys) == 0 and len(extra_keys) == 0:
        print("✓ Checkpoint is compatible with model architecture")
        return True
        
    print("\nCompatibility Report:")
    
    if len(mismatched_layers) > 0:
        print("\nMismatched layers:")
        for key, checkpoint_shape, model_shape in mismatched_layers:
            print(f"Layer: {key}")
            print(f"  Checkpoint shape: {checkpoint_shape}")
            print(f"  Model shape: {model_shape}")
    
    if len(missing_keys) > 0:
        print("\nKeys missing from checkpoint:")
        for key in missing_keys:
            print(f"  {key}")
    
    if len(extra_keys) > 0:
        print("\nExtra keys in checkpoint:")
        for key in extra_keys:
            print(f"  {key}")
            
    return False

if __name__ == "__main__":
    model_clip, _ = clip.load("ViT-B/32")
    model = Text_IF(model_clip)
    checkpoint_path = "path/to/your/checkpoint.pth"
    check_weights_compatibility(model, checkpoint_path)