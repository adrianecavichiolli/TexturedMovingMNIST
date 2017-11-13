"""
    mmnist_plus.py
    ~~~~~~~~~~~~~~

    Dataset class for Moving MNIST plus.
    
    Generates infinite random examples. 
    Some code adapted from https://gist.github.com/tencia/afb129122a64bde3bd0c

    @author: Lluis Castrejon
    @date: 2017-11-07
"""
# Imports ---------------------------------------------------------------------
import os
import sys
import math
import numpy as np

from copy import copy
from PIL import Image
from torch.utils.data import Dataset
# -----------------------------------------------------------------------------


# Set random seed
random_seed = 1337
rng = np.random.RandomState(random_seed)
data_dir = 'put_your_data_dir_here'
textures_dir = 'put_your_textures_dir_here'


def load_dataset():
    """
    Loads MNIST from the original web on demand.

    Returns:
        The MNIST dataset as a numpy np.float32 array n_examples x 1 x 28 x 28.
    """
    import gzip
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, out_path, source='http://yann.lecun.com/exdb/mnist/'):
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        print("Downloading %s" % filename)
        urlretrieve(source + filename, out_path)

    def load_mnist_images(dirname, filename):
        """
        Load the original MNIST images.
        """
        local_path = os.path.join(dirname, filename)
        if not os.path.exists(local_path):
            download(filename, local_path)

        with gzip.open(local_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        data = data/255.

        return data

    return load_mnist_images(data_dir,
                             'train-images-idx3-ubyte.gz')


def get_texture_patch(text_id, img_side, downsample=0):
    """
    Extract a random patch of texture #text_id from Brodatz dataset.

    Args:
        text_id: ID of the texture image to load.
        img_side: Side (as an integer) of the image to generate.
        downsample: Downsampling factor for the texture (integer).

    Returns:
        Random crop of the texture as a numpy array.
    """
    # Define fname of the texture
    fname = os.path.join(textures_dir, 'D{}.gif'.format(text_id))

    # Load image
    img_i = Image.open(fname)

    # Downsample image
    if downsample:
        img_i = img_i.resize((img_i.size[0]/downsample,
                              img_i.size[1]/downsample), Image.BILINEAR)

    # Extract random patch
    x = rng.randint(0, img_i.size[0] - img_side)
    y = rng.randint(0, img_i.size[1] - img_side)
    patch = img_i.crop((x, y, x + img_side, y + img_side))
    # patch = np.array(patch)

    return patch


def emboss(img, azi=45., ele=18., dep=2):
    """
    Perform embossing of image `img`.
    :param img: numpy.ndarray, matrix representing image to emboss.
    :param azi: azimuth (in degrees)
    :param ele: elevation (in degrees)
    :param dep: depth, (0-100)
    """
    # defining azimuth, elevation, and depth
    ele = (ele * 2 * np.pi) / 360.
    azi = (azi * 2 * np.pi) / 360.

    a = np.asarray(img).astype(np.float)
    # find the gradient
    grad = np.gradient(a)
    # (it is two arrays: grad_x and grad_y)
    grad_x, grad_y = grad
    # getting the unit incident ray
    gd = np.cos(ele)  # length of projection of ray on ground plane
    dx = gd * np.cos(azi)
    dy = gd * np.sin(azi)
    dz = np.sin(ele)
    # adjusting the gradient by the "depth" factor
    # (I think this is how GIMP defines it)
    grad_x = grad_x * dep / 100.
    grad_y = grad_y * dep / 100.
    # finding the unit normal vectors for the image
    leng = np.sqrt(grad_x**2 + grad_y**2 + 1.)
    uni_x = grad_x/leng
    uni_y = grad_y/leng
    uni_z = 1./leng
    # take the dot product
    a2 = 255 * (dx*uni_x + dy*uni_y + dz*uni_z)
    # avoid overflow
    a2 = a2.clip(0, 255)
    # you must convert back to uint8 /before/ converting to an image
    return Image.fromarray(a2.astype(np.uint8))


def emboss_img(img, patch_arr): 
    """
    Emboss an image.

    Based on make_mnistplus.py and taking some functions from that file.

    Args:
        img: Foreground image.
        patch_arr: Background image.
        img_size: Side of the image.

    Returns:
        The foregroung image is embossed in the background image.
    """
    # Generate binary mask that outlines the digits
    mask_arr = img > 0.1

    # Copy contents of masked-MNIST image into background texture
    blend_arr = copy(patch_arr)
    blend_arr[mask_arr] = img[mask_arr]

    # This the image to emboss
    frgd_img = Image.fromarray(blend_arr*255.)

    img = emboss(frgd_img)

    return img


class MMNISTPlus(Dataset):
    """
    Moving MNIST + dataset class.
    Generates a random example each time __getitem__ is called.
    """
    def __init__(self, seq_len, img_size, split, num_digits=1, 
                 fix_motion=False):
        """Initialize the dataset.
        
        Args:
            seq_len: Sequence length.
            img_size: Side of the image (they are square images).
            split: Split to use [train/val].
            num_digits: Number of digits to show in the image.
            fix_motion: Numbers always move in the same motion.
        """
        self.seq_len = seq_len
        self.img_size = img_size
        self.num_digits = num_digits
        self.fix_motion = fix_motion

        # Load MNIST dataset
        self.mnist = load_dataset()

    def _generate_example(self):
        """Generate one random example."""

        # Define limits of the image
        mnist_side = 28
        lims = (x_lim, y_lim) = (self.img_size - mnist_side,
                                 self.img_size - mnist_side)
        example = np.zeros((self.seq_len, self.img_size, self.img_size),
                           np.uint8)

        # Select a random texture. Note that texture 14 does not exist.
        texture_id = 14
        n_textures = 113
        while texture_id == 14:
            texture_id = rng.randint(1, n_textures)

        # Obtain a background image from the texture
        texture_img = get_texture_patch(texture_id, self.img_size)
        texture_arr = np.array(texture_img)

        # Note: we need texture array to be in float32 format
        #   This is because the way emboss was coded.
        texture_arr = texture_arr.astype(np.float)/255.

        # Select random MNIST digits
        mnist_images = rng.randint(0, self.mnist.shape[0], self.num_digits)
        mnist_images = [(self.mnist[i, 0].transpose()*255).astype(np.uint8) for i in mnist_images]
        mnist_images = [Image.fromarray(i) for i in mnist_images]

        # Select random starting positions for the MNIST digits
        positions = [(rng.rand()*x_lim, rng.rand()*y_lim)
                     for _ in xrange(self.num_digits)]

        # Select directions and velocity
        if not self.fix_motion:
            direcs = np.pi*(rng.rand(self.num_digits)*2 - 1)
        else:
            direcs = np.zeros(self.num_digits)

        if not self.fix_motion:
            speeds = rng.randint(5, size=self.num_digits)+2
        else:
            speeds = np.ones(self.num_digits) + 2

        veloc = [(v*math.cos(d), v*math.sin(d))
                 for d, v in zip(direcs, speeds)]

        # Generate frames
        for frame_idx in xrange(self.seq_len):

            # Create canvases for the numbers
            canvases = [Image.new('L', (self.img_size, self.img_size))
                        for _ in xrange(self.num_digits)]

            # Create canvas for the resulting image
            canvas = np.zeros((self.img_size, self.img_size), dtype=np.float32)

            # Paste numbers in the image canvas
            for i, canv in enumerate(canvases):
                canv.paste(mnist_images[i],
                           tuple(map(lambda p: int(round(p)), positions[i])))
                canvas += np.array(canv).astype(np.float)/255.

            # Update positions based on velocity
            next_pos = [map(sum, zip(p, v)) for p, v in zip(positions, veloc)]

            # Bounce off wall if a we hit one
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j+1:]))
            positions = [map(sum, zip(p, v)) for p, v in zip(positions, veloc)]

            # Copy additive canvas to data array
            example[frame_idx] = (canvas*255).astype(np.uint8).clip(0, 255)
            example[frame_idx] = emboss_img(example[frame_idx], texture_arr)


        return example

    def __getitem__(self, item):
        return self._generate_example().astype(np.float32)[:, None, :, :] 
    
    def __len__(self):
        return 10000  # This can be modified to any length
