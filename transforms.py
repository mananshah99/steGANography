def rotate_left_90(img):
    return img.transpose(-2, -1).flip(-2)

def add_gaussian_noise(img):
    return torch.clamp(img + torch.randn_like(img) * 2 - 1, -1, 1)

def color_filter(img, alpha=0.5, color=(255., 255., 255.)):
    cf = torch.tensor(color) / 255 * 2 - 1
    return (img.permute(0, 2, 3, 1) * (1 - alpha) + alpha * cf).permute(0, 3, 1, 2)
