from collections import deque

import numpy as np


def grow_region(img, seeds, threshold=65, up=True, down=True):
    """
    Find a region of connected pixels. The pixel values are assumed to be between 0 and 255
    https://en.wikipedia.org/wiki/Region_growing
    :param img: 2D numpy array
    :param seeds: list of (x, y) tuples
    :param threshold: the maximum difference between the seed and the grown pixels
    :param up: direction to consider when checking for neighbours
    :param down: direction to consider when checking for neighbours
    :return:
    """
    height = len(img)
    width = len(img[0])

    # Initialize 2D matrix
    grown = np.zeros((height, width), dtype=np.uint8)

    for pixel_position in seeds:
        x = pixel_position[0]
        y = pixel_position[1]
        grown[y, x] = 255

    for seed in seeds:
        queue = deque([seed])

        x = seed[0]
        y = seed[1]
        seed_intensity = img[y, x]
        seed_threshold = max(0, seed_intensity - threshold)
        while queue:
            pixel_position = queue.popleft()
            neighbours = get_neighbours(pixel_position, width, height, grown, up, down)
            for neighbour in neighbours:
                neighbour_x = neighbour[0]
                neighbour_y = neighbour[1]
                neighbour_intensity = img[neighbour_y, neighbour_x]
                if neighbour_intensity >= seed_threshold:
                    grown[neighbour_y, neighbour_x] = 255
                    queue.append(neighbour)

    return grown


def get_neighbours(position, width, height, grown, up, down):
    """
    Return connected neighbours that are not already "grown" (i.e. added to the region)
    """
    x = position[0]
    y = position[1]
    neighbours = []

    # Left
    if x > 0 and not grown[y, x - 1]:
        neighbours.append((x - 1, y))

    # Up
    if up and y > 0 and not grown[y - 1, x]:
        neighbours.append((x, y - 1))

    # Right
    if x < width - 1 and not grown[y, x + 1]:
        neighbours.append((x + 1, y))

    # Down
    if down and y < height - 1 and not grown[y + 1, x]:
        neighbours.append((x, y + 1))

    return neighbours
