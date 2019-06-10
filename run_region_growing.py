import numpy as np
import pyximport
from numpy.testing import assert_array_equal

pyximport.install(setup_args={'include_dirs': np.get_include()})
from PIL import Image

from region_growing import grow_region
from region_growing_cython import grow_region_cython
from timer import timer


if __name__ == "__main__":
    image = Image.open("example.png")
    image_np = np.array(image)
    seeds = [(200, 200)]

    with timer("region growing"):
        for _ in range(4):
            grown_image_up = grow_region(
                image_np, seeds, threshold=0, up=True, down=False
            )
            grown_image_down = grow_region(
                image_np, seeds, threshold=0, up=False, down=True
            )

    with timer("region growing cython"):
        for _ in range(4):
            grown_image_up_cython = grow_region_cython(
                image_np, seeds, threshold=0, up=True, down=False
            )
            grown_image_down_cython = grow_region_cython(
                image_np, seeds, threshold=0, up=False, down=True
            )

    assert_array_equal(grown_image_up, grown_image_up_cython)
    assert_array_equal(grown_image_down, grown_image_down_cython)

    Image.fromarray(grown_image_down).save("example_grown.png")
