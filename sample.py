import argparse
from matplotlib.image import imread, imsave
from models import *
import glob
from tqdm import tqdm
from scipy.misc import toimage

def _get_steganogan(path):
    steganogan_kwargs = {
        'cuda': False,
        'verbose': True,
        'path': path
    }

    return SteganoGAN.load(**steganogan_kwargs)

def generate_samples(model, cover, n, size=1):
    cover = torch.FloatTensor(cover) #.cuda()
    h = list(cover.shape)[0]
    w = list(cover.shape)[1]

    cover = cover.view(1, 3, h, w)

    generated = []
    for i in range(n):
        payload_i = model._random_data(cover, data_size=size) #.cuda()
        generated_i = model.encoder(cover, payload_i)[0].clamp(-1.0, 1.0)
        generated.append((generated_i.view(h, w, 3).cpu().detach().numpy() + 1.0) * 127.5)

    return generated

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-p', '--path')
    parser.add_argument('-d', '--original_dataset')
    parser.add_argument('-o', '--output_directory')
    parser.add_argument('-n', '--num_cover_images', type=int)
    parser.add_argument('-r', '--num_generated_per_cover', type=int)
    parser.add_argument('-s', '--size', type=float)

    args = parser.parse_args()

    model = _get_steganogan(args.path)

    # Generate samples

    pbar = tqdm(total = args.num_cover_images * args.num_generated_per_cover)

    num_processed = 0
    for filename in glob.glob(args.original_dataset + '*'):
        if num_processed > args.num_cover_images: break

        samples = generate_samples(model, imread(filename, pilmode='RGB') / 127.5 - 1.0, args.num_generated_per_cover, size = args.size)
        for i, sample in enumerate(samples):
            toimage(sample.astype('uint8')).save(args.output_directory + '/' + filename.split('/')[-1].split('.')[0] + '_sample_' + str(i) + '.png')
            pbar.update(1)
        
        num_processed += 1
    
    pbar.close()
main()
