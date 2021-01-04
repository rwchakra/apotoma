import numpy as np

ats = np.random.rand(60000) # Norms
threshold = 0.1


indexes = np.arange(ats.shape[0])
is_available = np.ones(shape=ats.shape[0], dtype=bool)

# This index used in the loop indicates the latest element selected to be added to chosen items
current_idx = 0

# for logging only
num_steps = 0

while True:
    num_steps += 1
    # Get all indexes (higher than current_index) which are still available and the corresponding ats
    candidate_indexes = np.argwhere((indexes > current_idx) & is_available).flatten()
    candidates = ats[candidate_indexes]
    print(f"candidates shape = {candidates.shape}")

    # Calculate the diff between norms (only this has to be slightly changed for norm of diffs impl)
    diffs = np.abs(candidates - ats[current_idx])
    assert diffs.ndim == 1

    # Identify candidates which are too similar to currently added element (current_idx)
    # and set their availability to false
    remove_candidate_indexes = np.flatnonzero(diffs < threshold)
    remove_overall_indexes = candidate_indexes[remove_candidate_indexes]
    is_available[remove_overall_indexes] = False

    # Select the next available candidate as current_idx (i.e., use select it for use in dsa),
    #   or break if none available
    if np.any(is_available[current_idx:]):
        current_idx = np.argmax(is_available[current_idx+1:]) + (current_idx + 1)
    else:
        # np.argmax did not find anything
        break


selected_indexes = np.nonzero(is_available)[0]
selected_ats = ats[selected_indexes]

print(f"num steps {num_steps}")
print(f"selected {selected_ats.shape[0]} ats: {selected_ats}")

# Test that indeed all differences are >= than thresholds

as_grid = np.expand_dims(selected_ats, axis=0).repeat(axis=0, repeats=selected_ats.shape[0])
grid_diffs = np.abs(as_grid - as_grid.T)
# Diagonals are 0 (subtraction with itself), replace with inf to allow check that all diffs are greater than thresholds
np.fill_diagonal(grid_diffs, np.inf)
assert np.all(grid_diffs > threshold)
