import torch

if torch.cuda.is_available():
    print(f"CUDA is available with {torch.cuda.device_count()} GPU(s).")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
else:
    print("CUDA is not available. Falling back to CPU.")