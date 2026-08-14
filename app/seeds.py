"""Where seeds come from, and why they are smaller than the column that holds them."""

import random

# The largest integer JavaScript represents exactly. Seeds are drawn below it.
#
# A seed is only worth recording if it can be read back and used again: the job page shows it and
# the create dialog accepts it, and reproducing a result by copying that number across is the
# entire point of storing it. JSON has no integer type — every number arrives in the browser as a
# double — so anything above 2**53 is silently rounded on the way in. The seed on screen would
# not be the seed that generated the video, and reproducing from it would quietly produce
# something else, with nothing anywhere reporting a problem.
#
# Postgres holds these in a BigInteger and could take the full 2**63. It is the display side that
# cannot, and nine quadrillion possible seeds is not a meaningful loss.
JS_SAFE_MAX_SEED = 2**53 - 1


def new_seed() -> int:
    """A fresh random seed, safe to show and to type back in."""
    return random.randint(0, JS_SAFE_MAX_SEED)
