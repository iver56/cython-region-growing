import numpy as np
import pyximport

pyximport.install()
from PIL import Image

from region_growing import grow_region
from region_growing_cython import grow_region_cython
from timer import timer


if __name__ == "__main__":
    image = Image.open("example.png")
    image_np = np.array(image)
    seeds = [(200, 200)]

    with timer("warmup"):
        grown_image_up = grow_region(image_np, seeds, threshold=0, up=True, down=False)
        grown_image_down = grow_region(
            image_np, seeds, threshold=0, up=False, down=True
        )

    with timer("region growing"):
        for _ in range(4):
            grown_image_up = grow_region(
                image_np, seeds, threshold=0, up=True, down=False
            )
            grown_image_down = grow_region(
                image_np, seeds, threshold=0, up=False, down=True
            )

    with timer("warmup"):
        grown_image_up = grow_region_cython(
            image_np, seeds, threshold=0, up=True, down=False
        )
        grown_image_down = grow_region_cython(
            image_np, seeds, threshold=0, up=False, down=True
        )

    with timer("region growing cython"):
        for _ in range(4):
            grown_image_up = grow_region_cython(
                image_np, seeds, threshold=0, up=True, down=False
            )
            grown_image_down = grow_region_cython(
                image_np, seeds, threshold=0, up=False, down=True
            )

    Image.fromarray(grown_image_down).save("example_grown.png")
