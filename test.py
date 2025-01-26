import GPUtil

gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU: {gpu.name}, Load: {gpu.load * 100}%, Memory Free: {gpu.memoryFree}MB")
