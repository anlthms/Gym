# ARC-AGI resource server

The resource server keeps expected outputs in session-owned state and exposes
deterministic training/test verification to the transform-refinement agent. It
validates grids, reports exactness and cell-level diffs, and finalizes the
episode without serializing hidden targets into model requests.

`create_dataset.py` converts ARC-AGI source data into NeMo-Gym rows. Existing
dataset releases use a single-test compatibility schema; the server normalizes
that schema when seeding a refinement session.
