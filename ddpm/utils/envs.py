# Copyright (c) Lornatang. All rights reserved.
import os
# Set and limit the number of threads used, from a minimum of 1 to a maximum of 8, and no more than the number of CPU cores minus 1
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))