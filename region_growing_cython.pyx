from collections import deque

import numpy as np
cimport numpy as np

def grow_region_cython(np.ndarray[np.uint8_t, ndim=2] img, seeds, threshold=65, up=True, down=True):
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
    cdef int height = len(img)
    cdef int width = len(img[0])
    cdef np.ndarray[np.uint8_t, ndim=2] grown = np.zeros((height, width), dtype=np.uint8)
    cdef int x
    cdef int y
    cdef int neighbour_x
    cdef int neighbour_y
    cdef int seed_intensity
    cdef int seed_threshold
    cdef int neighbour_intensity

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
            x = pixel_position[0]
            y = pixel_position[1]

            # Left
            if x > 0 and not grown[y, x - 1]:
                neighbour_x = x - 1
                neighbour_y = y
                neighbour_intensity = img[neighbour_y, neighbour_x]
                if neighbour_intensity >= seed_threshold:
                    grown[neighbour_y, neighbour_x] = 255
                    queue.append((neighbour_x, neighbour_y))

            # Up
            if up and y > 0 and not grown[y - 1, x]:
                neighbour_x = x
                neighbour_y = y - 1
                neighbour_intensity = img[neighbour_y, neighbour_x]
                if neighbour_intensity >= seed_threshold:
                    grown[neighbour_y, neighbour_x] = 255
                    queue.append((neighbour_x, neighbour_y))

            # Right
            if x < width - 1 and not grown[y, x + 1]:
                neighbour_x = x + 1
                neighbour_y = y
                neighbour_intensity = img[neighbour_y, neighbour_x]
                if neighbour_intensity >= seed_threshold:
                    grown[neighbour_y, neighbour_x] = 255
                    queue.append((neighbour_x, neighbour_y))

            # Down
            if down and y < height - 1 and not grown[y + 1, x]:
                neighbour_x = x
                neighbour_y = y + 1
                neighbour_intensity = img[neighbour_y, neighbour_x]
                if neighbour_intensity >= seed_threshold:
                    grown[neighbour_y, neighbour_x] = 255
                    queue.append((neighbour_x, neighbour_y))

    return grown
