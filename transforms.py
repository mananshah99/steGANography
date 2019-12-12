import torch

def rotate_left_90(img):
    return torch.rot90(img, 1, [2, 3])

def add_gaussian_noise(img):
    return torch.clamp(img + torch.randn_like(img) * 2 - 1, -1, 1)

def color_filter(img, alpha=0.5, color=(255., 255., 255.)):
    cf = torch.tensor(color, device=img.device) / 255 * 2 - 1
    filtered = (img.permute(0, 2, 3, 1) * (1 - alpha) + alpha * cf).permute(0, 3, 1, 2)
    return torch.clamp(filtered, -1, 1)
