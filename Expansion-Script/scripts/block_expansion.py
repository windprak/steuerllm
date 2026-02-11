import argparse
from transformers import AutoModelForCausalLM
import torch
import yaml

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path", default='meta-llama/Llama-2-7b-hf', type=str, help="original model path")
    parser.add_argument("--output_path", default='pytorch_model.bin', type=str, help="deepened model ckpt save path")
    parser.add_argument("--yaml_path", default='unfrozen_layers.yaml', type=str, help="path to save yaml file with unfrozen layers")
    parser.add_argument("--original_layers", default=32, type=int, help="original model num layers")
    parser.add_argument("--layers", default=40, type=int, help="deepen model num layers")

    # Parse the arguments
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    ckpt = model.state_dict()
    
    split = int(args.original_layers / (args.layers - args.original_layers))
    layer_cnt = 0

    output = {}
    unfrozen_layers = []

    for i in range(args.original_layers):
        for k in ckpt:
            if ('layers.' + str(i) + '.') in k:
                new_key = k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))
                output[new_key] = ckpt[k]
                print(f"Copied weight from original layer {i} to new layer {layer_cnt}: {new_key}")
        
        layer_cnt += 1
        if (i+1) % split == 0:
            for k in ckpt:
                if ('layers.' + str(i) + '.') in k:
                    new_key = k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))
                    if 'down_proj' in k or 'o_proj' in k:
                        output[new_key] = torch.zeros_like(ckpt[k])
                        print(f"Added zero weight for new layer {layer_cnt}: {new_key}")
                    else:
                        output[new_key] = ckpt[k]
                        print(f"Copied weight from original layer {i} to new layer {layer_cnt}: {new_key}")
                    
                    # Collect unfrozen layer names for YAML file
                    unfrozen_layers.append(new_key)
            layer_cnt += 1
        
    assert layer_cnt == args.layers
    for k in ckpt:
        if not 'layers' in k:
            output[k] = ckpt[k]

    # Save the updated model
    torch.save(output, args.output_path)
    print(f"Model saved to {args.output_path}")

    # Generate the YAML file
    yaml_data = {
        "unfrozen_parameters": [
            "^lm_head.weight$",
            "^model.embed_tokens.weight$",
        ]
    }
    for layer in unfrozen_layers:
        yaml_data["unfrozen_parameters"].append(f"^{layer}$")

    with open(args.yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    print(f"Unfrozen layers saved to {args.yaml_path}")

if __name__ == "__main__":
    main()
