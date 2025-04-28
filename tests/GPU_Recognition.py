try:
    import torch

    print("Torch version:", torch.__version__)
    print("CUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available. Running on CPU.")

except ImportError:
    print("Torch is not installed. Please install it to check GPU availability.")
