#!/usr/bin/env python3
import argparse
from transformers import AutoModelForCausalLM
import torch

def main():
    parser = argparse.ArgumentParser(description="Debug model keys")
    parser.add_argument("--model_path", required=True, type=str, help="model path")
    args = parser.parse_args()
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    ckpt = model.state_dict()
    
    print(f"\nTotal keys in state dict: {len(ckpt.keys())}")
    
    # Find language model layer keys
    language_model_keys = [k for k in ckpt.keys() if 'language_model' in k and 'layers.' in k]
    print(f"Language model layer keys found: {len(language_model_keys)}")
    
    # Show first 10 language model keys
    print("\nFirst 10 language model layer keys:")
    for i, key in enumerate(language_model_keys[:10]):
        print(f"  {i+1}: {key}")
    
    # Check for layer 0 specifically
    layer_0_keys = [k for k in ckpt.keys() if 'layers.0.' in k]
    print(f"\nLayer 0 keys found: {len(layer_0_keys)}")
    for key in layer_0_keys[:5]:
        print(f"  {key}")
    
    # Check for vision tower keys
    vision_keys = [k for k in ckpt.keys() if 'vision_tower' in k]
    print(f"\nVision tower keys found: {len(vision_keys)}")
    for key in vision_keys[:5]:
        print(f"  {key}")

if __name__ == "__main__":
    main()
