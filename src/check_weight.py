"""
This code developed by @wearetyomsmnv. Commercial use - prohibited.
"""

import argparse
import struct
import hashlib
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel

# Правила для обнаружения потенциально вредоносного кода
MALICIOUS_STRINGS = [
    "os", "runpy", "builtins", "ccommands", "subprocess", "c__builtin__",
    "execvp", "popen", "call", "Popen", "check_call", "run", "eval", "exec", "compile", "open", "run_code",
    "requests", "pip install", "posix", "nt"
]

def check_malicious_code(content: str):
    """
    Проверяет содержимое на наличие потенциально вредоносных команд и библиотек.
    Возвращает True, если найдено совпадение с известными вредоносными строками.
    """
    for pattern in MALICIOUS_STRINGS:
        if pattern in content:
            print(f"Potentially malicious code detected: {pattern}")
            return True
    return False

def pytorch_steganography(model_path: Path, payload: Path, n=3):
    assert 1 <= n <= 8

    # Load model
    model = torch.load(model_path, map_location=torch.device("cpu"))

    # Read the payload
    size = os.path.getsize(payload)

    with open(payload, "rb") as payload_file:
        message = payload_file.read()

    # Payload data layout: size + sha256 + data
    payload = struct.pack("i", size) + bytes(hashlib.sha256(message).hexdigest(), "utf-8") + message

    # Get payload as bit stream
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))

    if (len(bits) % n) != 0:
        # Pad bit stream to multiple of bit count
        bits = np.append(bits, np.full(shape=n-(len(bits) % n), fill_value=0, dtype=bits.dtype))

    bits_iter = iter(bits)

    for item in model:
        tensor = model[item].data.numpy()

        # Ensure the data will fit
        if np.prod(tensor.shape) * n < len(bits):
            continue

        print(f"Hiding message in layer {item}...")

        # Bit embedding mask
        mask = 0xff
        for i in range(0, tensor.itemsize):
            mask = (mask << 8) | 0xff

        mask = mask - (1 << n) + 1

        # Create a read/write iterator for the tensor
        with np.nditer(tensor.view(np.uint32), op_flags=["readwrite"]) as tensor_iterator:
            # Iterate over float values in tensor
            for f in tensor_iterator:
                # Get next bits to embed from the payload
                lsb_value = 0
                for i in range(0, n):
                    try:
                        lsb_value = (lsb_value << 1) + next(bits_iter)
                    except StopIteration:
                        assert i == 0

                        # Save the model back to disk
                        torch.save(model, f=model_path)

                        return True

                # Embed the payload bits into the float
                f = np.bitwise_and(f, mask)
                f = np.bitwise_or(f, lsb_value)

                # Update the float value in the tensor
                tensor_iterator[0] = f

    return False

def check_hidden_content(tensor, n=3):
    assert 1 <= n <= 8

    tensor_data = tensor.numpy()
    
    # Bit mask for extracting the embedded bits
    mask = (1 << n) - 1

    # Iterate over float values in tensor
    hidden_bits = []
    with np.nditer(tensor_data.view(np.uint32), flags=["refs_ok"], op_flags=["readwrite"]) as tensor_iterator:
        for f in tensor_iterator:
            # Extract the least significant bits (LSBs)
            lsb_value = np.bitwise_and(f, mask)
            hidden_bits.append(lsb_value)

    # Convert extracted bits back to bytes
    hidden_bits = np.array(hidden_bits, dtype=np.uint8)
    hidden_bytes = np.packbits(hidden_bits)
    
    # Attempt to interpret the first part of the data as the size and hash
    try:
        extracted_size = struct.unpack("i", hidden_bytes[:4])[0]
        extracted_hash = hidden_bytes[4:68].decode("utf-8")
        extracted_message = hidden_bytes[68:68+extracted_size]
        
        # Calculate the hash of the extracted message
        calculated_hash = hashlib.sha256(extracted_message).hexdigest()
        
        if calculated_hash == extracted_hash:
            print(f"Hidden content found in tensor!")
            print(f"Extracted Size: {extracted_size}")
            print(f"Extracted SHA256 Hash: {extracted_hash}")
            print("The hidden content matches the embedded payload.")
            return True
        else:
            print(f"Tensor checked: No matching hidden content found.")
    
    except Exception as e:
        print(f"Error while checking tensor: {e}")

    print("No hidden content detected in the tensor.")
    return False

def check_weights_and_hidden_content(model_name, layer_name=None, n=3):
    try:
        # Load model from Huggingface
        model = AutoModel.from_pretrained(model_name)

        # Check specific layer if provided
        if layer_name:
            if layer_name in model.state_dict():
                weights = model.state_dict()[layer_name]
                print(f"Checking layer: {layer_name}")
                has_nan = torch.isnan(weights).any().item()
                has_inf = torch.isinf(weights).any().item()

                print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")
                if not has_nan and not has_inf:
                    print("Checking for hidden content in this layer...")
                    check_hidden_content(weights, n)
                    print("Checking for potentially malicious code...")
                    if check_malicious_code(weights.numpy().tobytes().decode('utf-8', errors='ignore')):
                        print("Warning: Potentially malicious code found in this layer.")
            else:
                print(f"Layer '{layer_name}' not found in the model.")
        else:
            # Check all layers if no specific layer is provided
            print("Checking all layers...")
            for name, param in model.named_parameters():
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                if has_nan or has_inf:
                    print(f"Problem in layer {name}: Has NaN - {has_nan}, Has Inf - {has_inf}")
                else:
                    print(f"Layer {name} is clean. Checking for hidden content...")
                    check_hidden_content(param, n)
                    print("Checking for potentially malicious code...")
                    if check_malicious_code(param.numpy().tobytes().decode('utf-8', errors='ignore')):
                        print(f"Warning: Potentially malicious code found in layer {name}.")

    except Exception as e:
        print(f"Error while checking model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Steganography, Hidden Content Checker, and Weight Validator")
    parser.add_argument("action", choices=["embed", "check_weights"], help="Action to perform: 'embed' to hide content or 'check_weights' to validate weights and check for hidden content.")
    parser.add_argument("model", type=Path, help="Path to the model file or Huggingface model name for weight validation.")
    parser.add_argument("--payload", type=Path, help="Path to the payload file to embed (required for 'embed' action).")
    parser.add_argument("--bits", type=int, choices=range(1, 9), default=3, help="Number of bits used for embedding/checking (default: 3).")
    parser.add_argument("--layer_name", type=str, help="Specific layer name to check for NaN/Inf values and hidden content (for 'check_weights' action).")

    args = parser.parse_args()

    if args.action == "embed":
        if args.payload is None:
            parser.error("'embed' action requires --payload argument.")
        else:
            if pytorch_steganography(args.model, args.payload, n=args.bits):
                print("Embedded payload in model successfully.")
            else:
                print("Failed to embed payload in the model.")
    elif args.action == "check_weights":
        check_weights_and_hidden_content(args.model, args.layer_name, args.bits)