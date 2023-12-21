import numpy as np
import logging

logger = logging.getLogger(__name__)

np.seterr(all="warn")

# Overflow
a = np.array([1e200])
b = np.array([1e200])

# Raise warning
c = a * b

try:
    assert c <= 1e6
except AssertionError:
    # Make very big instead: kludge
    c = 1e6
    logger.warning("Corrected c")

print(f"{c=}")
