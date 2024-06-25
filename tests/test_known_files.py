# assert 
# - number of vsync blocks
# - total number of diode flips

import time
import numpy as np
import npc_sync


def test_file_with_aborted_stim_block() -> None:
    """A stimulus was aborted before any data was saved, but it has a short stim
    running block with one vsync which can cause problems if not removed from our
    filtered listof stim running edges."""
    s = npc_sync.SyncDataset(
        's3://aind-private-data-prod-o5171v/ecephys_726088_2024-06-18_12-58-00/behavior/20240618T125800.h5'
    )
    assert len(s.stim_running_edges[0]) == 7

def test_standard_file() -> None:
    """A standard file with no issues."""
    s = npc_sync.SyncDataset('s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5')
    assert len(s.stim_running_edges[0]) == 7
    vsync_times_in_blocks = s.vsync_times_in_blocks
    frame_display_time_blocks = s.frame_display_time_blocks
    assert len(vsync_times_in_blocks) == 7
    assert len(frame_display_time_blocks) == 7
    assert len(np.concatenate(vsync_times_in_blocks)) == 414763
    assert len(np.concatenate(frame_display_time_blocks)) == 414763
    
if __name__ == "__main__":
    import pytest
    pytest.main(['-v', __file__])