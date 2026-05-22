import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import webdataset as wds
from PIL import Image, ImageDraw
from glitch_this import ImageGlitcher
from scipy import spatial
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from custom_dataset import UnstructuredImageDataset

# Cap the number of tiles loaded from the tile bank.
MAX_TILES_TO_LOAD = 5000


def build_output_filename(
    input_filename: str,
    augment_type: str,
    intensity_level: int,
    keep_original_name: bool = False,
) -> str:
    """Build the output filename for a generated sample.

    - Default: use robust suffix naming like `image_mosaic_1.JPEG`.
    - If keep_original_name=True: keep the filename aligned with the source sample name.

    Using Path.stem avoids the old `name[:-4]` bug for `.jpeg` files.
    """
    input_path = Path(input_filename)
    if keep_original_name:
        return input_path.name

    base_name = input_path.stem
    return f"{base_name}_{augment_type}_{intensity_level}.JPEG"


def identity_function(x):
    return x
    

def transforms_geometric_shapes(intensity_level):
    """
    Creates a composition of transforms to apply geometric shapes augmentation to images.

    Parameters:
    - num_tiles (int): Number of geometric shapes to add to each image.

    Returns:
    - torchvision.transforms.Compose: A composition of geometric shape image transforms.
    """
    if intensity_level==1:
        num_tiles=150
    elif intensity_level==2:
        num_tiles=300
    elif intensity_level==3:
        num_tiles=600
    elif intensity_level==4:
        num_tiles=800
    elif intensity_level==5:
        num_tiles=1000
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224),PasteShapes(num_tiles),transforms.ToTensor()],
    )
    return transform

def draw_vertical_lines(im,intensity_level):
    """
    Draws vertical lines of an image based on the specified intensity level.
    
    Parameters:
    - im (PIL.Image): The input image based on which to draw vertical lines.
    - intensity_level (int): the intensity level of the distortion

    Returns:
    - PIL.Image: The image with vertical line.
    """
    intensity_mapping = {
        1: {"l": 224, "step": 1},
        2: {"l": 178, "step": 2},
        3: {"l": 112, "step": 4},
        4: {"l": 84, "step": 6},
        5: {"l": 60, "step": 8}
        }
    l= intensity_mapping[intensity_level]["l"]
    step=intensity_mapping[intensity_level]["step"]
    c1 = (0, 0, 0)
    c2 = (200, 200, 200)
    im1 = im.convert('L')
    im2 = Image.new("RGB", im.size, color=c1)
    draw = ImageDraw.Draw(im2)
    _x, _y = 0, 0
    w, h = im.size
    for i in range(l):
        x = (i / l) * w
        _x, _y = x, 0
        grid_x_start = int(x)
        grid_x_end = min(int(x + w / l), w)

        for y in range(0, h, step):
            grid_y_start = y
            grid_y_end = min(y + step, h)
            sum_r = sum_g = sum_b = 0
            count = 0
            sum_intensity=0
            for grid_x in range(grid_x_start, grid_x_end):
                for grid_y in range(grid_y_start, grid_y_end):
                    r, g, b = im.getpixel((grid_x, grid_y))
                    intensity = im1.getpixel((grid_x, grid_y))
                    sum_intensity += intensity
                    sum_r += r
                    sum_g += g
                    sum_b += b
                    count += 1
            if count > 0:
                avg_r = sum_r // count
                avg_g = sum_g // count
                avg_b = sum_b // count
                avg_color = (avg_r, avg_g, avg_b)
                avg_intensity = sum_intensity // count
            else:
                avg_color = (0, 0, 0)
                avg_intensity = 0
            xEff = 1 - avg_intensity / 255
            xEff *= -40
            xy = (_x, _y, x + xEff, y)
            draw.line(xy, fill=avg_color)
            _x, _y = x + xEff, y

    return im2
    

def transforms_default(tile_size):
    """
    Creates a default transform composition for image preprocessing without specific augmentations.

    Parameters:
    - tile_size (int): Size to resize and center crop the images to.

    Returns:
    - torchvision.transforms.Compose: A composition of default image transforms.
    """
    transform = transforms.Compose(
            [transforms.Resize(256-256%tile_size), transforms.CenterCrop(224-224%tile_size),transforms.ToTensor()]
        )
    return transform
    
def transforms_mosaic(tile_size):
    """
    Creates a default transform composition for image preprocessing without specific augmentations.

    Parameters:
    - tile_size (int): Size to resize and center crop the images to.

    Returns:
    - torchvision.transforms.Compose: A composition of default image transforms.
    """
    if tile_size==6:
        transform = transforms.Compose(
            [transforms.Resize(256-256%tile_size), transforms.CenterCrop(252),transforms.ToTensor()]
        )
    else:
        transform = transforms.Compose(
                [transforms.Resize(256-256%tile_size), transforms.CenterCrop(224),transforms.ToTensor()]
            )
    return transform

def transforms_images(tile_size,intensity_level,augment_type):
    """
    Selects and applies the appropriate image transformations based on the geometric shapes augmentation type.

    Parameters:
    - tile_size (int): The tile size for image transformation.
    - num_tiles (int): The number of tiles or shapes to use in the augmentation.
    - augment_type (str): The type of augmentation to apply.

    Returns:
    - torchvision.transforms.Compose: A composition of selected image transforms.
    """
    if augment_type=='geometric_shapes':
        return transforms_geometric_shapes(intensity_level)
    elif augment_type in ['sticker','luminance','glitched','vertical_lines']:
        return transforms_default(tile_size=16)
    elif augment_type=='mosaic':
        return transforms_mosaic(tile_size)
    

def get_dataloader(
    datadir: str,
    tile_size: int = None,
    batch_size: int = 256,
    train_mode: bool = True,
    augment_type: str = "",
    intensity_level: str = "",
):
    """Create a dataloader for tile-bank loading or image corruption."""
    if train_mode:
        if tile_size is None:
            raise ValueError("tile_size must not be None if train_mode=True.")

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize((tile_size, tile_size)),
                transforms.ToTensor(),
            ]
        )
        dataset = (
            wds.WebDataset(datadir, empty_check=False)
            .decode("pil")
            .to_tuple("jpg cls.txt __key__")
            .map_tuple(transform, identity_function, identity_function)
            .batched(batch_size, partial=True)
        )

        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
        )
    else:
        transform = transforms_images(tile_size, intensity_level, augment_type)
        dataset = UnstructuredImageDataset(datadir, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=8,
        )

    return dataloader    


shape_list = ['triangle', 'square', 'star', 'circle']
def random_color():
    return tuple(np.random.randint(0, 256, size=3))

def random_size():
    return np.random.randint(6, 16)

def random_rotation():
    return np.random.randint(0, 360)

def draw_shape(draw, shape, position, size, color, rotation):
    """
    Draws a geometric shape on an image.

    Parameters:
    - draw (ImageDraw.Draw): Drawing context for a PIL image.
    - shape (str): Type of shape to draw (e.g., 'triangle', 'square', 'star', 'circle').
    - position (tuple): The (x, y) coordinates for the shape's starting position.
    - size (int): The size of the shape. For non-circular shapes, this typically refers to the side length or a related dimension.
    - color (tuple): The color of the shape, defined as a (R, G, B) tuple.
    - rotation (int): Rotation angle of the shape in degrees, applicable for shapes like 'star' where rotation makes a visual difference.

    Returns:
    - None.
    """
    if shape == 'triangle':
        points = [(0, 0), (size, 0), (size / 2, size)]
        draw.polygon([(p[0] + position[0], p[1] + position[1]) for p in points], fill=color)
    elif shape == 'square':
        draw.rectangle([position, (position[0] + size, position[1] + size)], fill=color)
    elif shape == 'star':
        outer_radius = size/1.2
        inner_radius = outer_radius / 2
        num_points = 5  # Five-pointed star
        points = []
        for i in range(num_points * 2):
            angle = math.radians(i * 180 / num_points + rotation)
            radius = outer_radius if i % 2 == 0 else inner_radius
            points.append((position[0] + radius * math.sin(angle), position[1] + radius * math.cos(angle)))
        draw.polygon(points, fill=color)
    elif shape == 'circle':
        draw.ellipse([position, (position[0] + size, position[1] + size)], fill=color)
        
def paste_shapes(image, num_shapes):
    """
    Pastes a specified number of random geometric shapes onto an image.

    Parameters:
    - image (PIL.Image): Image on which to paste shapes.
    - num_shapes (int): Number of shapes to paste onto the image.

    Returns:
    - PIL.Image: Image with pasted shapes.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for _ in range(num_shapes):
        shape = np.random.choice(shape_list)
        position = (np.random.randint(0, width), np.random.randint(0, height))
        size = random_size()
        color = random_color()
        rotation = random_rotation()
        draw_shape(draw, shape, position, size, color, rotation)

    return image
    
class PasteShapes:
    def __init__(self, num_shapes):
        self.num_shapes = num_shapes

    def __call__(self, img):
        return paste_shapes(img.copy(), self.num_shapes)

def randomize_illumination(tile,intensity_level,i,j):
    """
    Randomly adjusts the illumination of a given image tile based on a specified intensity level.

    Parameters:
    - tile (np.ndarray): Image tile to adjust.
    - intensity_level (int): Level of intensity for the adjustment, determines the range of brightness adjustment.
    - i (int): Row index of the tile.
    - j (int): Column index of the tile.

    Returns:
    - np.ndarray: Image tile with adjusted illumination.
    """
    if intensity_level==1:
        brightness_adjustment = np.random.uniform(0, 50)*(-1)**(i+j)
    elif intensity_level==2:
        brightness_adjustment = np.random.uniform(50, 100)*(-1)**(i+j)
    elif intensity_level==3:
        brightness_adjustment=np.random.uniform(100, 125)*(-1)**(i+j)
    elif intensity_level==4:
        brightness_adjustment=np.random.uniform(125, 150)*(-1)**(i+j)
    elif intensity_level==5:
        brightness_adjustment=np.random.uniform(150, 255)*(-1)**(i+j)
    
    if tile.dtype == np.float32:
        
        brightness_adjustment /= 255.0
        augmented_tile = np.clip(tile + brightness_adjustment, 0, 1)
    return augmented_tile

def prepare_tiles(train_dataloader) -> np.ndarray:
    """Load a capped number of tiles for sticker augmentation."""
    tiles = []
    total_loaded = 0
    dataloader_iter = iter(train_dataloader)
    num_tiles_to_load = MAX_TILES_TO_LOAD

    print(f"Preparing tiles for 'sticker' augmentation (limit: {num_tiles_to_load} tiles)...")
    try:
        while total_loaded < num_tiles_to_load:
            tiles_batch, _, _ = next(dataloader_iter)

            if total_loaded + len(tiles_batch) > num_tiles_to_load:
                needed = num_tiles_to_load - total_loaded
                tiles_batch = tiles_batch[:needed]

            tiles_batch_np = tiles_batch.numpy()
            tiles.append(tiles_batch_np)
            total_loaded += len(tiles_batch)

            print(f"Tiles loaded so far: {total_loaded}/{num_tiles_to_load}")
    except StopIteration:
        print("Reached the end of the tile dataset.")

    if len(tiles) > 0:
        tiles = np.concatenate(tiles, axis=0)
        print("Final tiles shape for sticker:", tiles.shape)
    else:
        print("No tiles were loaded.")
        tiles = np.array([])

    return tiles

def prepare_tiles_mosaic(
    train_dataloader,
    tile_size: int,
) -> Tuple[np.ndarray, np.ndarray, spatial.KDTree]:
    """Load a capped number of tiles and build the KD-Tree for mosaic augmentation."""
    num_tiles_to_load = MAX_TILES_TO_LOAD
    tiles_list = []
    total_loaded = 0

    print(f"Preparing tiles for 'mosaic' augmentation (limit: {num_tiles_to_load} tiles)...")
    for _, (x, _, _) in enumerate(train_dataloader):
        if total_loaded >= num_tiles_to_load:
            print(f"Reached tile limit of {num_tiles_to_load}. Stopping tile loading.")
            break

        if total_loaded + len(x) > num_tiles_to_load:
            needed = num_tiles_to_load - total_loaded
            x = x[:needed]

        tiles_list.append(x.numpy())
        total_loaded += len(x)
        print(f"Tiles loaded so far: {total_loaded}/{num_tiles_to_load}")

    if not tiles_list:
        print("[Error] No tiles were loaded for mosaic.")
        return np.array([]), None

    tiles = np.concatenate(tiles_list, axis=0)
    print("Final tiles shape for mosaic:", tiles.shape)

    print("Calculating mean colors for KD-Tree...")
    keys = tiles.mean((-2, -1))
    print("Building KD-Tree...")
    tree = spatial.KDTree(keys)
    print("KD-Tree built successfully.")

    return tiles, tree
 
def vertical_lines_augmentation(x,augment_type,intensity_level):
    """
    Applies geometric shapes augmentation to a dataset.

    Parameters:
    - x (np.ndarray): The input images.
    - augment_type (str): Specifies the type of augmentation ('geometric_shapes').
    - num_tiles (int): The number of geometric shapes to add to each image.

    Returns:
    - str: Folder name where augmented images are stored.
    - np.ndarray: The augmented images.
    """
    enclosing_folder=f'{augment_type}_dataset/intensity_level_{intensity_level}/'
    return enclosing_folder,x


def geometric_shapes_augmentation(x,augment_type,num_tiles):
    """
    Applies geometric shapes augmentation to a dataset.

    Parameters:
    - x (np.ndarray): The input images.
    - augment_type (str): Specifies the type of augmentation ('geometric_shapes').
    - num_tiles (int): The number of geometric shapes to add to each image.

    Returns:
    - str: Folder name where augmented images are stored.
    - np.ndarray: The augmented images.
    """
    enclosing_folder=f'{augment_type}_dataset/shape_num_{num_tiles}/'
    return enclosing_folder,x
  
  
def luminance_augmentation(x,intensity_level,augment_type):
    """
    Adjusts the luminance of image tiles based on specified intensity levels.

    Parameters:
    - x (np.ndarray): The input images.
    - tile_size (int): Size of each tile for luminance adjustment.
    - intensity_level (int): Intensity level for luminance adjustment.
    - augment_type (str): Specifies the type of luminance augmentation.

    Returns:
    - str: Folder name where augmented images are stored.
    - np.ndarray: The images with adjusted luminance.
    """
    tile_size=14
    for i, iy in enumerate(range(0, x.shape[2], tile_size)):
        for j, ix in enumerate(range(0, x.shape[3], tile_size)):
            current_block = x[:, :, iy : iy + tile_size, ix : ix + tile_size]
            augmented_block = randomize_illumination(current_block,intensity_level,i,j)
            x[:, :, iy : iy + tile_size, ix : ix + tile_size] = augmented_block
    enclosing_folder=f'{augment_type}_dataset_checkerboard/tile_size_{tile_size}_intensity_level_{intensity_level}/'
    return enclosing_folder,x


   
def sticker_augmentations(x,augment_type,tiles,batch_size,intensity_level):
    """
    Applies sticker augmentations to images, overlaying them with random tiles.

    Parameters:
    - x (np.ndarray): The input images.
    - tile_size (int): The size of each tile to overlay.
    - augment_type (str): Specifies the sticker augmentation type.
    - tiles (np.ndarray): The tiles to overlay on images.
    - batch_size (int): The number of images in each batch.
    - num_tiles (int): The number of tiles to overlay on each image.

    Returns:
    - str: Folder name where augmented images are stored.
    - np.ndarray: The images with stickers applied.
    """
    tile_size=16
    if intensity_level==1:
        num_tiles=100
    elif intensity_level==2:
        num_tiles=200
    elif intensity_level==3:
        num_tiles=400
    elif intensity_level==4:
        num_tiles=600
    elif intensity_level==5:
        num_tiles=1200
    if augment_type in ['sticker','sticker_single_class']:
        tile_indices =  np.random.choice(len(tiles),  (batch_size,num_tiles), replace=True)
        selected_tiles = tiles[tile_indices.reshape(-1)].reshape(batch_size, num_tiles, 3, tile_size, tile_size)
    print('got tiles!')
    image_height, image_width = x.shape[2:]
    availability_mask = np.ones((batch_size, image_height, image_width), dtype=bool)
    availability_mask[:, -tile_size:, :] = False
    availability_mask[:, :, -tile_size:] = False
    tile_pos= np.random.choice(image_height-tile_size, size=(batch_size,num_tiles, 2), replace=True)
    for i, iy in enumerate(range(0, batch_size)):
        for j, ix in enumerate(range(0,num_tiles)):
            available_positions = np.argwhere(availability_mask[i])
            if len(available_positions) > 0:
                selected_idx = np.random.randint(0, len(available_positions))
                selected_position = available_positions[selected_idx]
            else:
                print("No available positions left.")
                break
            tile_pos = selected_position
            if augment_type in ['sticker','sticker_single_class']:
                x[i, :, tile_pos[0]:tile_pos[0] + tile_size, tile_pos[1]:tile_pos[1] + tile_size] = selected_tiles[i, j]
            elif augment_type =='random_maskedout':
                x[i, :, tile_pos[0]:tile_pos[0] + tile_size, tile_pos[1]:tile_pos[1] + tile_size] = 0*x[i, :, tile_pos[0]:tile_pos[0] + tile_size, tile_pos[1]:tile_pos[1] + tile_size]
    if augment_type =='sticker':
        enclosing_folder=f'{augment_type}_dataset_overlapping/tile_size_{tile_size}_tile_num_{num_tiles}/'
    return enclosing_folder,x
    


def glitched(x,intensity_level,augment_type):
    """
    Applies a glitch effect to images based on the specified intensity level.

    Parameters:
    - x (np.ndarray): The input images.
    - intensity_level (int): The level of glitch intensity, which affects how strong the glitch effect will be.
    - augment_type (str): Specifies the type of augmentation ('glitched').

    Returns:
    - str: Folder name where augmented images are stored.
    - np.ndarray: The augmented images with glitch effects applied.
    """
    glitcher = ImageGlitcher()
    intensity_mapping = {
        1: {"glitch_level": 2},
        2: {"glitch_level": 4},
        3: {"glitch_level": 5},
        4: {"glitch_level": 8},
        5: {"glitch_level": 10},
    }
    glitch_level= intensity_mapping[intensity_level]["glitch_level"]
    corrupted = (x * 255).astype(np.uint8).transpose((0, 2, 3, 1))
    to_pil = ToPILImage()
    for i in range(corrupted.shape[0]):
        img_pil = to_pil(corrupted[i])
        glitch_img = glitcher.glitch_image(img_pil,glitch_level,
                                           glitch_change=0.0,
                                           cycle=False,
                                           scan_lines=True,
                                           color_offset=True,
                                           seed=1,
                                           gif=False,
                                           frames=23,
                                           step=1)
        corrupted[i] = np.array(glitch_img)
    corrupted = np.transpose(corrupted/255, (0, 3, 1, 2))
    enclosing_folder=f'{augment_type}_dataset/intensity_level_{intensity_level}/'
    return enclosing_folder,corrupted


def augmentation_process(x, augment_type, intensity_level, batch_size, tiles):
    """Dispatch to the selected augmentation."""
    operations = {
        "sticker": lambda x: sticker_augmentations(x, augment_type, tiles, batch_size, intensity_level),
        "luminance": lambda x: luminance_augmentation(x, intensity_level, augment_type),
        "geometric_shapes": lambda x: geometric_shapes_augmentation(x, augment_type, intensity_level),
        "glitched": lambda x: glitched(x, intensity_level, augment_type),
        "vertical_lines": lambda x: vertical_lines_augmentation(x, augment_type, intensity_level),
    }

    if augment_type in operations:
        return operations[augment_type](x)

    print("Augmentation type not supported")
    return None, x
        
    

def process_data(
    test_dataloader,
    tile_size: int,
    tiles: np.ndarray,
    target_folder: str,
    intensity_level: int,
    augment_type: str,
    keep_original_name: bool = False,
):
    """Apply the selected augmentation and save the corrupted images."""
    to_tensor = transforms.ToTensor()
    to_pil = ToPILImage()

    for x, p in tqdm(test_dataloader):
        print("Started batch")
        x = [to_pil(xi) for xi in x]
        current_x = torch.stack([to_tensor(xi) for xi in x])
        print("Current image tensor shape:", current_x.shape)

        x = current_x.numpy()
        batch_size = x.shape[0]

        print("Starting augmentation...")
        _, x = augmentation_process(x, augment_type, intensity_level, batch_size, tiles)

        x = (x.transpose((0, 2, 3, 1)) * 255).clip(0, 255).astype(np.uint8)
        os.makedirs(target_folder, exist_ok=True)

        for xi, pi in zip(x, p):
            img = Image.fromarray(xi)
            if augment_type == "vertical_lines":
                img = draw_vertical_lines(img, intensity_level)

            new_p = os.path.join(
                target_folder,
                build_output_filename(pi, augment_type, intensity_level, keep_original_name),
            )
            os.makedirs(os.path.dirname(new_p), exist_ok=True)
            img.save(new_p, format="JPEG", subsampling=0, quality=100)
            
def process_data_mosaic(
    test_dataloader,
    tile_size: int,
    tree: spatial.KDTree,
    tiles: np.ndarray,
    target_folder: str,
    intensity_level: int,
    augment_type: str,
    keep_original_name: bool = False,
):
    """
    Applies mosaic augmentation to a dataset using a spatial KDTree for nearest neighbor tile matching.

    Parameters:
    - test_dataloader (Loader): The dataloader containing the images to be augmented.
    - tile_size (int): The size of the tiles used for the mosaic.
    - tree (spatial.KDTree): KDTree used for nearest-neighbor search to find the best matching tiles.
    - tiles (np.ndarray): Pre-prepared tiles for the mosaic.
    - target_folder (str): Directory where the augmented images will be saved.
    - intensity_level (int): The intensity level of the augmentation.
    - augment_type (str): The type of augmentation to apply, specifically 'mosaic' in this case.

    Notes:
    - This function processes images in batches, applies mosaic-like tiling based on nearest neighbors in the KDTree, and saves the augmented images to the specified directory.
    """
    
    
    to_tensor = transforms.ToTensor()
    to_pil = ToPILImage()
    def center_crop(image, target_size=(224, 224)):
        width, height = image.size
        new_width, new_height = target_size
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        return image.crop((left, top, right, bottom))

    for x, p in tqdm(test_dataloader):
        x = [to_pil(xi) for xi in x]
        resized_x = [
            xi.resize((xi.size[0] // tile_size, xi.size[1] // tile_size)) for xi in x
        ]
        resized_x = [to_tensor(xi) for xi in resized_x]
        resized_x = torch.stack(resized_x)
        resized_x = resized_x.permute((0, 2, 3, 1)).contiguous()
        x = torch.stack([to_tensor(xi) for xi in x]).numpy()

        orig_resized_x_shape = resized_x.shape
        resized_x = resized_x.view(-1, 3).numpy()
        best_tile_indices = tree.query(resized_x)[1]
        best_tile_indices = best_tile_indices.reshape(orig_resized_x_shape[:3])
        best_tiles = tiles[best_tile_indices]
        for i, iy in enumerate(range(0, x.shape[2], tile_size)):
            for j, ix in enumerate(range(0, x.shape[3], tile_size)):
                x[:, :, iy : iy + tile_size, ix : ix + tile_size] = best_tiles[:, i, j]
        x = (x.transpose((0, 2, 3, 1)) * 255).clip(0, 255).astype(np.uint8)
        print('x_finished!')
        enclosing_folder_path = target_folder
        print('folder path created!')
        if not os.path.exists(enclosing_folder_path):
            os.makedirs(enclosing_folder_path,exist_ok=True)
        for xi, pi in zip(x, p):
            img = Image.fromarray(xi)
            if tile_size==6:
                img = center_crop(img, (224, 224))
            new_p = os.path.join(
                target_folder,
                build_output_filename(pi, augment_type, intensity_level, keep_original_name),
            )
            print('new_p',new_p)
            os.makedirs(os.path.dirname(new_p), exist_ok=True)
            img.save(new_p, format="JPEG", subsampling=0, quality=100)
            print('img_saved!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_folder", type=str)
    parser.add_argument("--target_folder", type=str)
    parser.add_argument(
        "--tile_dir",
        type=str,
        default=str(Path(os.environ.get("DATA_ROOT", "/mnt/d/stamp_lib/datasets")) / "imagenet_val_tiles.tar"),
    )
    parser.add_argument(
        "--augment_type",
        type=str,
        choices=[
            "geometric_shapes",
            "sticker",
            "luminance",
            "glitched",
            "vertical_lines",
            "mosaic",
        ],
        help="Augmentation type.",
    )
    parser.add_argument("--intensity_level", type=int, default=0)
    parser.add_argument(
        "--keep_original_name",
        action="store_true",
        help="Save outputs with the exact same filename as the source image.",
    )
    args = parser.parse_args()

    tiles = []
    if args.augment_type == "sticker":
        tile_size = 16
        print("Building tile loader for sticker augmentation...")
        train_dataloader = get_dataloader(
            args.tile_dir,
            intensity_level=args.intensity_level,
            tile_size=tile_size,
            augment_type=args.augment_type,
        )
        tiles = prepare_tiles(train_dataloader)
    elif args.augment_type == "mosaic":
        if args.intensity_level == 1:
            tile_size = 4
        elif args.intensity_level == 2:
            tile_size = 6
        elif args.intensity_level == 3:
            tile_size = 8
        elif args.intensity_level == 4:
            tile_size = 16
        elif args.intensity_level == 5:
            tile_size = 28

        train_dataloader = get_dataloader(
            args.tile_dir,
            tile_size,
            augment_type=args.augment_type,
        )
        tiles, tree = prepare_tiles_mosaic(train_dataloader, tile_size)
    else:
        tile_size = 16

    print("Building input loader...")
    test_dataloader = get_dataloader(
        os.path.join(args.source_folder),
        intensity_level=args.intensity_level,
        train_mode=False,
        tile_size=tile_size,
        augment_type=args.augment_type,
    )

    if args.augment_type == "mosaic":
        process_data_mosaic(
            test_dataloader,
            tile_size,
            tree,
            tiles,
            args.target_folder,
            args.intensity_level,
            args.augment_type,
            keep_original_name=args.keep_original_name,
        )
    else:
        process_data(
            test_dataloader,
            tile_size,
            tiles,
            args.target_folder,
            args.intensity_level,
            args.augment_type,
            keep_original_name=args.keep_original_name,
        )


if __name__ == "__main__":
    main()
