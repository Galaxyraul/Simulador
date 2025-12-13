from torch import device
def check_resources(N:int):
    RAM_THRESHOLD = 0.8   # 80% of system RAM
    GPU_THRESHOLD = 0.8   # 80% of GPU memory

    # --- Fields and dtypes ---
    dtype_sizes = {
        'uint8': 1,
        'float16': 2,
        'float32': 4
    }

    fields = {
        'state': 'uint8',
        'days_in_state': 'uint8',
        'times_infected': 'uint8',
        'susceptibility': 'float16',
        'noncompliance': 'float16',
        'mobility': 'float16',
        'age_factor': 'float16'
    }

    # --- Compute estimated memory per shard ---
    total_bytes = N * sum(dtype_sizes[dtype] for dtype in fields.values())
    total_MB = total_bytes / 1e6
    total_GB = total_bytes / 1e9
    print(f"Estimated memory per shard: {total_MB:.2f} MB ({total_GB:.2f} GB)")

    # --- Check system RAM ---
    mem = psutil.virtual_memory()
    available_RAM_GB = mem.available / 1e9
    if total_GB > available_RAM_GB * RAM_THRESHOLD:
        raise MemoryError(f"Shard requires {total_GB:.2f} GB, which exceeds "
                        f"{RAM_THRESHOLD*100}% of available system RAM ({available_RAM_GB:.2f} GB).")

    print(f"Available system RAM: {available_RAM_GB:.2f} GB - OK")

    # --- Check GPU memory if available ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
        total_GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_GPU_GB = torch.cuda.memory_allocated(0) / 1e9
        free_GPU_GB = total_GPU_GB - allocated_GPU_GB

        if total_GB > free_GPU_GB * GPU_THRESHOLD:
            raise MemoryError(f"Shard requires {total_GB:.2f} GB, which exceeds "
                            f"{GPU_THRESHOLD*100}% of free GPU memory ({free_GPU_GB:.2f} GB).")
        
        print(f"Free GPU memory: {free_GPU_GB:.2f} GB - OK")
    return total_GB > available_RAM_GB * RAM_THRESHOLD