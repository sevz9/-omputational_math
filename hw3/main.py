import argparse
from PIL import Image
import numpy as np
from read_write import write_compressed, read_compressed, write_image
import scipy.linalg as la

def numpy_svd(channel):
    return np.linalg.svd(channel)


def simple_svd(channel, max_iters = 100, tol = 0.001):
    U = np.zeros((channel.shape))
    S = np.zeros(channel.shape[0])
    V = np.zeros((channel.shape))

    for _ in range(max_iters):
        Q, R = np.linalg.qr(channel @ V)
        U = Q
        Q, R = np.linalg.qr(channel.T @ U)
        V = Q
        S = np.diag(R)
        if np.linalg.norm(channel @ V - U @ R) < tol:
            break 
   
    return U, S, V.T

def advanced_svd(channel, k):
    U = np.zeros((channel.shape[0], k))
    S = np.zeros(k)
    Vt = np.zeros((k, channel.shape[1]))

    for i in range(k):
        x = np.random.normal(0, 1, size=channel.shape[1])
        x = channel.T @ channel @ x

        v = x / np.linalg.norm(x)
        sigma = np.linalg.norm(channel @ v)
        u = channel @ v / sigma

        U[:, i] = u
        Vt[i, :] = v
        S[i] = sigma
        channel = channel - sigma * np.outer(u, v)

    return U, S, Vt

def compress_image(in_file, out_file, method, compression):
    image = Image.open(in_file)
    channels = np.array(image)
    new_channels = []

    m, n, channels_num = channels.shape
    channels = [channels[:, :, i] for i in range(channels_num)]
    k = int(m * n / ((m + n) * compression))
 
    for channel in channels:
        if method  == 'numpy':
            U, S, Vh = numpy_svd(channel)
            new_channels.append((U[:, :k], S[:k], Vh[:k, :]))
        elif method == 'simple':
            U, S, Vh = simple_svd(channel)
            new_channels.append((U[:, :k], S[:k], Vh[:k, :]))
        elif method == 'advanced':
            U, S, Vh = advanced_svd(channel, k)
            new_channels.append((U, S, Vh))
        else:
            raise Exception(f"no method {method}")
        
    write_compressed(out_file, new_channels, m, n, k)
    

def decompress_image(in_file, out_file):
    m, n, k, new_channels = read_compressed(in_file)
    write_image(out_file, m, n, k, new_channels)


def main():
    parser = argparse.ArgumentParser(description="Compress or decompress an image.")
    parser.add_argument("--mode", type=str, choices=["compress", "decompress"], required=True)
    parser.add_argument("--method", type=str, choices=["numpy", "simple", "advanced"])
    parser.add_argument("--compression", type=int)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "compress":
        compress_image(args.in_file, args.out_file, args.method, args.compression)
    elif args.mode == "decompress":
        decompress_image(args.in_file, args.out_file)

if __name__ == "__main__":
    main()
