import struct
import numpy as np
from PIL import Image

def write_compressed(path, channels, m, n, k):
    with open(path, "wb") as f:
        f.write(struct.pack("<III", m, n, k))

        for channel in channels:
            U, S, Vh = channel
            f.write(U.astype(np.float32).flatten().tobytes())
            f.write(S.astype(np.float32).flatten().tobytes())
            f.write(Vh.astype(np.float32).flatten().tobytes())


def read_compressed(path, NUMBER_OF_CHANNELS=3):
    with open(path, "rb") as f:
      
        m, n, k = struct.unpack("<III", f.read(3 * 4))
        channels = []
        for _ in range(NUMBER_OF_CHANNELS):
            U = np.frombuffer(f.read(m * k * 4), dtype=np.float32).reshape(
                (m, k)
            )
            
            S = np.frombuffer(f.read(k * 4), dtype=np.float32)
            Vh = np.frombuffer(f.read(k * n * 4), dtype=np.float32).reshape(
                (k, n)
            )
            channels.append((U, S, Vh))
  
    return m, n, k, channels

def write_image(path, m, n, k, svd_channels):
    channels = []
    for svd_channel in svd_channels:
        U, S, Vh = svd_channel
        channels.append(U @ np.diag(S) @ Vh)
    
    arr = np.array(channels)
    arr = np.stack(arr, axis=2)
 
    arr = arr.astype(np.uint8)
  
    im = Image.fromarray(arr)
    im.save(path)