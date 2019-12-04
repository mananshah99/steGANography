from PIL import Image
from matplotlib.image import imread, imsave
import argparse
import torch
import torch.nn.functional as F

from models import *
from loader import *

def occlusion_heatmap(model, image, label, occ_size = 50, occ_stride = 50, occ_pixel = 0.5):
    width, height = list(image.shape)[-2], list(image.shape)[-1]
    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))
  
    heatmap = torch.zeros((output_height, output_width))
    
    for h in range(0, height):
        for w in range(0, width):
            
            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            
            if (w_end) >= width or (h_end) >= height:
                continue
            
            input_image = image.clone().detach()
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel
            
            output = model(input_image)
            print(output)
            output = F.softmax(output, dim=1)
            print(output)
            prob = output.tolist()[0][label]
            
            heatmap[h, w] = prob 
    return heatmap

def _get_steganogan(path):
    
    steganogan_kwargs = {
        'cuda': True,
        'verbose': True
    }
    steganogan_kwargs['path'] = path

    return SteganoGAN.load(**steganogan_kwargs)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str)
parser.add_argument('-i', '--image', type=str)
args = parser.parse_args()

steganogan = _get_steganogan(args.path)
critic = steganogan.critic
image = torch.FloatTensor(imread(args.image, pilmode='RGB') / 127.5 - 1.0).cuda()
image = image.permute(2, 1, 0).unsqueeze(0)

hm = occlusion_heatmap(critic, image, 0)
print(hm)
