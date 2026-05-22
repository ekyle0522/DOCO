# -*- coding: utf-8 -*-
"""Generate six OOD-C datasets with a single script.

Datasets:
- Places365-C
- iNaturalist-C
- SUN-C
- Textures-C
- SSB-Hard-C
- NINCO_OOD_classes-C

Usage:
- Run `python corruption_data_preparation.py` to generate all datasets
  listed in DATASETS_TO_PROCESS.
- To generate only one dataset, comment out the other names in
  DATASETS_TO_PROCESS.

Resume feature:
- The script supports breakpoint resume.
- Before each run, it checks whether the target output file already exists.
- Existing samples are skipped automatically, and only missing samples are
  generated.

Note:
- The `frost` corruption requires six template files stored at:
  ./frost_template/frost1.png ... frost6.jpg
"""

import ctypes
import os
import warnings
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import skimage as sk
import torch
from PIL import Image as PILImage
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

warnings.simplefilter("ignore", UserWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('DATA_ROOT', '/mnt/d/stamp_lib/datasets'))

DATASETS_TO_PROCESS = [
    'places365',
    'inaturalist',
    'sun',
    'textures',
    'ssb-hard',
    'ninco_ood_classes',
]

# Whether to skip samples whose target files already exist.
# Set to False if you want to regenerate everything.
SKIP_EXISTING = True



def disk(radius: int, alias_blur: float = 0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)

    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
)


class MotionImage(WandImage):
    def motion_blur(self, radius: float = 0.0, sigma: float = 0.0, angle: float = 0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)



def plasma_fractal(mapsize: int = 256, wibbledecay: float = 3):
    """Generate a heightmap using the diamond-square algorithm."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float32)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]

        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)

        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()



def clipped_zoom(img: np.ndarray, zoom_factor: float):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]



def ensure_pil_image(x):
    if isinstance(x, PILImage.Image):
        return x
    return PILImage.fromarray(np.uint8(np.clip(x, 0, 255)))



def get_frost_files() -> List[Path]:
    frost_files = [
        SCRIPT_DIR / 'frost_template' / 'frost1.png',
        SCRIPT_DIR / 'frost_template' / 'frost2.png',
        SCRIPT_DIR / 'frost_template' / 'frost3.png',
        SCRIPT_DIR / 'frost_template' / 'frost4.jpg',
        SCRIPT_DIR / 'frost_template' / 'frost5.jpg',
        SCRIPT_DIR / 'frost_template' / 'frost6.jpg',
    ]
    missing = [str(p) for p in frost_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing frost template files. Please place the following files under '
            'the current script directory:\n' + '\n'.join(missing)
        )
    return frost_files


FROST_FILES = get_frost_files()



def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255



def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255



def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255



def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255



def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(np.array(x) / 255., sigma=c)
    return np.clip(x, 0, 1) * 255



def glass_blur(x, severity=1):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    for _ in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255



def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255



def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    ensure_pil_image(x).save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())
    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.ndim == 2:
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
    return np.clip(x[..., [2, 1, 0]], 0, 255)



def zoom_blur(x, severity=1):
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255



def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255



def frost(x, severity=1):
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    idx = np.random.randint(len(FROST_FILES))
    frost_img = cv2.imread(str(FROST_FILES[idx]))
    x_start = np.random.randint(0, frost_img.shape[0] - 224)
    y_start = np.random.randint(0, frost_img.shape[1] - 224)
    frost_img = frost_img[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * frost_img, 0, 255)



def snow(x, severity=1):
    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())
    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(
        x,
        cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5,
    )
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255



def spatter(x, severity=1):
    c = [
        (0.65, 0.3, 4, 0.69, 0.6, 0),
        (0.65, 0.3, 3, 0.68, 0.6, 0),
        (0.65, 0.3, 2, 0.68, 0.5, 0),
        (0.65, 0.3, 1, 0.65, 1.5, 1),
        (0.67, 0.4, 1, 0.65, 1.5, 1),
    ][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0

    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.maximum(np.max(m, axis=(0, 1)), 1e-12)
        m *= c[4]

        color = np.concatenate(
            (
                175 / 255. * np.ones_like(m[..., :1]),
                238 / 255. * np.ones_like(m[..., :1]),
                238 / 255. * np.ones_like(m[..., :1]),
            ),
            axis=2,
        )

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)
        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255

    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0

    color = np.concatenate(
        (
            63 / 255. * np.ones_like(x[..., :1]),
            42 / 255. * np.ones_like(x[..., :1]),
            20 / 255. * np.ones_like(x[..., :1]),
        ),
        axis=2,
    )

    color *= m[..., np.newaxis]
    x *= (1 - m[..., np.newaxis])
    return np.clip(x + color, 0, 1) * 255



def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255



def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255



def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255



def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    output = BytesIO()
    ensure_pil_image(x).save(output, 'JPEG', quality=c)
    output.seek(0)
    return PILImage.open(output).convert('RGB')



def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = ensure_pil_image(x)
    x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x = x.resize((224, 224), PILImage.BOX)
    return x



def elastic_transform(image, severity=1):
    c = [
        (244 * 2, 244 * 0.7, 244 * 0.1),
        (244 * 2, 244 * 0.08, 244 * 0.2),
        (244 * 0.05, 244 * 0.01, 244 * 0.02),
        (244 * 0.07, 244 * 0.01, 244 * 0.02),
        (244 * 0.12, 244 * 0.01, 244 * 0.02),
    ][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size,
    ])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


FULL_CORRUPTIONS: Dict[str, Callable] = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'defocus_blur': defocus_blur,
    'glass_blur': glass_blur,
    'motion_blur': motion_blur,
    'zoom_blur': zoom_blur,
    'snow': snow,
    'frost': frost,
    'fog': fog,
    'brightness': brightness,
    'contrast': contrast,
    'elastic_transform': elastic_transform,
    'pixelate': pixelate,
    'jpeg_compression': jpeg_compression,
}


DATASET_CONFIGS = {
    'places365': {
        'base_folder': 'PLACES365',
        'output_dir': DATA_ROOT / 'PLACES365-C',
        'save_layout': 'flat',
        'corruptions': FULL_CORRUPTIONS,
        'batch_size': 128,
        'num_workers': 32,
    },
    'inaturalist': {
        'base_folder': 'iNaturalist',
        'output_dir': DATA_ROOT / 'iNaturalist-C',
        'save_layout': 'flat',
        'corruptions': FULL_CORRUPTIONS,
        'batch_size': 128,
        'num_workers': 32,
    },
    'sun': {
        'base_folder': 'SUN',
        'output_dir': DATA_ROOT / 'SUN-C',
        'save_layout': 'flat',
        'corruptions': FULL_CORRUPTIONS,
        'batch_size': 128,
        'num_workers': 32,
    },
    'textures': {
        'base_folder': 'Textures/images',
        'output_dir': DATA_ROOT / 'Textures-C',
        'save_layout': 'subfld',
        'corruptions': FULL_CORRUPTIONS,
        'batch_size': 128,
        'num_workers': 32,
    },
    'ssb-hard': {
        'base_folder': 'SSB-Hard',
        'output_dir': DATA_ROOT / 'SSB-Hard-C',
        'save_layout': 'subfld',
        'corruptions': FULL_CORRUPTIONS,
        'batch_size': 128,
        'num_workers': 32,
    },
    'ninco_ood_classes': {
        'base_folder': 'NINCO/NINCO_OOD_classes',
        'output_dir': DATA_ROOT / 'NINCO_OOD_classes-C',
        'save_layout': 'subfld',
        'corruptions': FULL_CORRUPTIONS,
        'batch_size': 128,
        'num_workers': 32,
    },
}


class CorruptionSaver(ImageFolder):
    def __init__(
        self,
        root: Path,
        base_folder: str,
        method: Callable,
        level: int,
        output_dir: Path,
        save_layout: str,
        skip_existing: bool = True,
        transform=None,
        **kwargs,
    ) -> None:
        self.root_dir = Path(root).expanduser()
        self.dataset_folder = self.root_dir / base_folder
        self.method = method
        self.level = level
        self.output_dir = Path(output_dir) / self.method.__name__ / str(self.level)
        self.save_layout = save_layout
        self.skip_existing = skip_existing

        if not self.dataset_folder.exists():
            raise FileNotFoundError(f'Input dataset directory does not exist: {self.dataset_folder}')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(str(self.dataset_folder), transform=transform, **kwargs)

        self.all_samples: List[Tuple[str, int]] = list(self.samples)
        self.total_count = len(self.all_samples)

        if self.skip_existing:
            pending_samples = []
            existing_count = 0
            for path, target in self.all_samples:
                if self.build_output_path(path).is_file():
                    existing_count += 1
                else:
                    pending_samples.append((path, target))

            self.samples = pending_samples
            self.imgs = self.samples
            self.targets = [target for _, target in self.samples]
            self.existing_count = existing_count
            self.pending_count = len(self.samples)
        else:
            self.existing_count = 0
            self.pending_count = self.total_count

    def build_output_path(self, input_path: str) -> Path:
        relative_path = Path(os.path.relpath(input_path, self.dataset_folder))
        if self.save_layout == 'flat':
            return self.output_dir / relative_path
        if self.save_layout == 'subfld':
            return self.output_dir / 'images' / relative_path
        raise ValueError(f'Unknown save_layout: {self.save_layout}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        output_path = self.build_output_path(path)

        if self.skip_existing and output_path.is_file():
            # Defensive check for rare race conditions when restarting or using workers.
            sample = self.loader(path)
            sample = transforms.Resize((224, 224))(sample)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target

        sample = self.loader(path)
        img_corrupted = transforms.Resize((224, 224))(sample)
        img_corrupted = self.method(img_corrupted, self.level)
        img_corrupted = ensure_pil_image(img_corrupted)
        assert img_corrupted.size == (224, 224)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_corrupted.save(output_path)

        if self.transform is not None:
            img_corrupted = self.transform(img_corrupted)
        return img_corrupted, target



def generate_one_dataset(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_CONFIGS:
        raise KeyError(f'Unsupported dataset: {dataset_name}')

    cfg = DATASET_CONFIGS[dataset_name]
    root = DATA_ROOT

    print('=' * 80)
    print(f'Start generating: {dataset_name}')
    print(f'Input directory: {root / cfg["base_folder"]}')
    print(f'Output directory: {cfg["output_dir"]}')
    print(f'Save layout: {cfg["save_layout"]}')
    print(f'Breakpoint resume: {"enabled" if SKIP_EXISTING else "disabled"}')
    print('=' * 80)

    # Keep only severity level 5.
    level = 5

    for domain, method in cfg['corruptions'].items():
        dataset = CorruptionSaver(
            root=root,
            base_folder=cfg['base_folder'],
            method=method,
            level=level,
            output_dir=cfg['output_dir'],
            save_layout=cfg['save_layout'],
            skip_existing=SKIP_EXISTING,
            transform=transforms.ToTensor(),
        )

        print(
            f'[{dataset_name} | {domain} | level {level}] '
            f'total={dataset.total_count}, existing={dataset.existing_count}, pending={dataset.pending_count}'
        )

        if dataset.pending_count == 0:
            print(f'[{dataset_name} | {domain} | level {level}] already finished, skip.')
            continue

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers'],
            drop_last=False,
            pin_memory=False,
            persistent_workers=cfg['num_workers'] > 0,
        )

        processed_now = 0
        total_batches = len(data_loader)

        for batch_idx, (_img, _target) in enumerate(data_loader, start=1):
            processed_now += len(_target)

            if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
                finished = dataset.existing_count + processed_now
                print(
                    f'[{dataset_name} | {domain} | level {level}] '
                    f'batch {batch_idx}/{total_batches}, '
                    f'finished {finished}/{dataset.total_count}'
                )

        print(f'[{dataset_name} | {domain} | level {level}] done.')

    print(f'Finished: {dataset_name}')



def main():
    for dataset_name in DATASETS_TO_PROCESS:
        generate_one_dataset(dataset_name)


if __name__ == '__main__':
    main()
