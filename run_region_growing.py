from PIL import Image
import numpy as np

from region_growing import grow_region
from timer import timer

if __name__ == "__main__":
    image = Image.open("example.png")
    image_np = np.array(image)
    seeds = [(200, 200)]

    with timer('region growing'):
        for _ in range(8):
            grown_image = grow_region(image_np, seeds, threshold=0)

    Image.fromarray(grown_image).save("example_grown.png")
