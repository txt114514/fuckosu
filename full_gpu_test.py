import torch
import time

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    size = 6000
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")

    torch.cuda.synchronize()
    t0 = time.time()

    c = torch.matmul(a, b)

    torch.cuda.synchronize()
    t1 = time.time()

    print("GPU computation success")
    print("Time:", t1 - t0)
else:
    print("GPU not available")
