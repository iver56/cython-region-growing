# Region growing algorithm implemented in Cython

Set up Cython and a C++ compiler and install the dependencies specified in `requirements.txt`.

Then run `python run_region_growing.py` to test the two region growing algorithms, one implemented in Python and the other in Cython.

Both execution time measurements and correctness checks are in place.

On my machine, the Cython variant is one magnitude faster:

* Python: 540 ms
* Cython: 31 ms
