import numpy as np
from collections import namedtuple
from typing import List
from numpy.linalg import inv


# Stores spatial and temporal gradients.
Gradient = namedtuple('Gradient', 'ix iy it')


# 2D velocity vector
Velocity = namedtuple('Velocity', 'x y')


# Represents a region of an image in frame t and frame t+1.
# Also stores the center of the region wrt the original image.
Region = namedtuple('Region', 'pixels next_pixels center_row center_col')


# Represents a motion estimate for a region.
MotionEstimate = namedtuple('MotionEstimate', 'velocity center_row center_col')


def gradient_x(block: np.array) -> float:
    """
    :param block: a 2x2 block of an image.
    :return: the spatial gradient in the x direction.
    """
    return (block[:,1] - block[:,0]).sum()


def gradient_y(block: np.array) -> float:
    """
    :param block: a 2x2 block of an image.
    :return: the spatial gradient in the y direction.
    """
    return (block[0,:] - block[1,:]).sum()


def gradient_t(block1: np.array, block2: np.array) -> float:
    """
    :param block1: a 2x2 block of image in frame t.
    :param block2: a 2x2 block of the corresponding section of image in frame t+1.
    :return: the spatial gradient in the time direction.
    """
    return (block2 - block1).sum()


def estimate_gradients(block1: np.array, block2: np.array) -> Gradient:
    """
    :param block1: a 2x2 block of image in frame t.
    :param block2: a 2x2 block of the corresponding section of image in frame t+1.
    :return: the spatial and temporal gradients for the block.
    """
    # Normalise by dividing by 4.
    ix = (gradient_x(block1) + gradient_x(block2)) / 4
    iy = (gradient_y(block1) + gradient_y(block2)) / 4
    it = gradient_t(block1, block2) / 4

    return Gradient(ix, iy, it)


def estimate_all_gradients(image_region: np.array, next_image_region: np.array) -> List[Gradient]:
    """
    Calculates the gradients for all 2x2 blocks in the image region.
    :param image_region: a region of image in frame t.
    :param next_image_region: the corresponding image region in frame t+1.
    :return: the gradients for all 2x2 blocks in the image region.
    """
    rows, cols = image_region.shape
    gradients = []

    # Get all 2x2 blocks in both regions and calculate the velocities for them.
    for min_row in range(0, rows - 1):
        for min_col in range(0, cols - 1):
            block1 = image_region[min_row:min_row+2, min_col:min_col+2]
            block2 = next_image_region[min_row:min_row+2, min_col:min_col+2]

            gradients.append(estimate_gradients(block1, block2))

    return gradients


def construct_matrices(block_gradients: List[Gradient]) -> (np.array, np.array):
    """
    Constructs the matrices A and b from the gradients of each block in an image segment.
    :param block_gradients: a list of gradients of each 2x2 block in an image segment.
    :return: a tuple containing matrices A and b.
    """
    A = np.zeros((2, 2))
    b = np.zeros((2))

    for ix, iy, it in block_gradients:
        A[0, 0] += ix**2
        A[0, 1] += ix * iy
        A[1, 0] += ix * iy
        A[1, 1] += iy**2

        b[0] -= ix * it
        b[1] -= iy * it

    return (A, b)


def calculate_velocity(A: np.array, b: np.array) -> Velocity:
    """
    :return: the velocity calculated using matrices A and b.
    """
    # There might be no inverse for A.
    try:
        v = inv(A).dot(b)
        return Velocity(v[0], v[1])
    except:
        return Velocity(0, 0)


def generate_image_regions(frame: np.array, next_frame: np.array, region_size: int) -> List[Region]:
    """
    :param frame: a frame of video at time t.
    :param next_frame: a frame of video at time t+1.
    :param region_size: the size to divide the frames into (to calculate motion for).
    :return: a list of regions
    """
    regions = []

    rows, cols = frame.shape

    # Ignore any regions that don't fit at the borders of the image.
    num_regions_rows = int(rows / region_size)
    num_regions_cols = int(cols / region_size)

    for i in range(0, num_regions_rows):
        for j in range(0, num_regions_cols):
            min_row, max_row = i * region_size, (i + 1) * region_size
            min_col, max_col = j * region_size, (j + 1) * region_size

            region_pixels = frame[min_row:max_row, min_col:max_col]
            next_region_pixels = next_frame[min_row:max_row, min_col:max_col]

            center_row = int((min_row + max_row) / 2)
            center_col = int((min_col + max_col) / 2)

            r = Region(pixels=region_pixels, next_pixels=next_region_pixels, center_row=center_row, center_col=center_col)
            regions.append(r)

    return regions


def estimate_region_motion(region: Region) -> MotionEstimate:
    """
    Estimates the motions a region
    :param region: the regions to estimate motion for.
    :return: the motion estimates for the regions.
    """
    gs = estimate_all_gradients(region.pixels, region.next_pixels)
    A, b = construct_matrices(gs)
    velocity = calculate_velocity(A, b)

    return MotionEstimate(velocity, region.center_row, region.center_col)


def estimate_motion(frame: np.array, next_frame: np.array, region_size: int) -> List[MotionEstimate]:
    """
    Estimates motions for all regions in the frames.
    :param frame: the image at frame t.
    :param next_frame: the image at frame t+1.
    :return: a list of motion estimates for each region.
    """
    regions = generate_image_regions(frame, next_frame, region_size)
    return [estimate_region_motion(r) for r in regions]
