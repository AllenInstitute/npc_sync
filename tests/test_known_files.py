# assert 
# - number of vsync blocks
# - total number of diode flips

import os
import time

import numpy as np
import pytest

import npc_sync


def test_screen_flicker_file() -> None:
    """One of the sessions that has 'phantom' screen flicker in the sync square at
    the start of some stim blocks. 
    
    - processing fails to divide diode flips into blocks correctly. If we simply divide based
      on stim running signal, the number of flips in each blocks is way off (both
      too many in some blocks, and too few in others).
    - the default behavior is to detect the mismatch with the number of vsyncs and
      return vsync time + constant monitor lag: to disable this behavior, set the
      env var DEBUG_SYNC to anything

    - the difference in number of diode edges vs vsyncs is so big there must be
      missing diode flips as well as extra ones. The extra ones are likely due to
      the screen flicker issue.. missing ones are harder to explain. 
    
    - conclusion: stick to vsync time + constant monitor lag
    """
    # os.environ['DEBUG_SYNC'] = '1' # uncomment to test properly
    s = npc_sync.SyncDataset(
        's3://aind-private-data-prod-o5171v/ecephys_706401_2024-04-24_12-26-03/behavior/20240424T122603.h5'
    )
    vsync_times_in_blocks = [51336, 12655, 36000, 36000, 216864, 36000, 36000, 12647]
    frames_in_stim_files = [51336, 12655, 36000, 36000, 216864, 36000, 36000, 12647]
    # vsyncs match stim files as expected
    for idx, frame_times in enumerate(s.frame_display_time_blocks):
        assert len(frame_times) == frames_in_stim_files[idx], f"block {idx}: {len(frame_times) = }, expected {frames_in_stim_files[idx] = }"

def test_slow_processing_file() -> None:
    """File with many missing diode flips, caused processing to be slow by trying
    to fill-in the missing diode flips. This is a rare case, but it's good to have
    a check for it.
    
    - was resolved by checking number of missing flips, and if above a threshold
      using vsync times + constant monitor lag
    """
    t0 = time.time()
    s = npc_sync.SyncDataset(
        's3://aind-private-data-prod-o5171v/ecephys_699847_2024-04-18_13-42-34/behavior/20240418T134234.h5'
    )
    _ = s.frame_display_time_blocks
    t = time.time() - t0
    assert t < 60, f"file with many missing diode flips is being processed too slowly: {t = }"


def test_missing_diode_flips_at_start_of_block() -> None:
    def helper() -> None:
        s = npc_sync.SyncDataset(
            's3://aind-ephys-data/ecephys_666986_2023-08-14_11-38-55/behavior/20230814T113855.h5'
        )
        _ = s.frame_display_time_blocks
    # launch job in a separate thread, then kill if it takes too long
    import threading
    t = threading.Thread(target=helper)
    t.start()
    t.join(20)
    assert not t.is_alive(), "file with missing diode flips at start of block is taking too long to process"


@pytest.mark.skip(reason="We have no way to resolve this issue yet, and so far it has only affected files other than the main DR task")
def test_vsync_mismatch_file() -> None:
    """A session that has stim blocks with vsyncs recorded on sync that were not
    recorded in the corresponding stim files"""
    s = npc_sync.SyncDataset(
        's3://aind-private-data-prod-o5171v/ecephys_703880_2024-04-18_13-06-56/behavior/20240418T130656.h5'
    )
    frames_in_stim_files = [51336, 18955, 36000, 36000, 216923, 36000, 36000, 18931]

    assert len(s.stim_running_edges[0]) == len(frames_in_stim_files)
    vsync_times_in_blocks = s.vsync_times_in_blocks
    # -> [51336, 18976, 36006, 36001, 216923, 36000, 36000, 18931]
    frame_display_time_blocks = s.frame_display_time_blocks
    assert len(vsync_times_in_blocks) == len(frames_in_stim_files)
    assert len(frame_display_time_blocks) == len(frames_in_stim_files)
    assert len(s.get_falling_edges('vsync_stim')) == sum(frames_in_stim_files), f"{len(s.get_falling_edges('vsync_stim')) = }, expected {sum(frames_in_stim_files) = }"
    assert len(np.concatenate(vsync_times_in_blocks)) == sum(frames_in_stim_files), f"{len(np.concatenate(vsync_times_in_blocks)) = }, expected {sum(frames_in_stim_files) = }"
    assert len(np.concatenate(frame_display_time_blocks)) == sum(frames_in_stim_files), f"{len(np.concatenate(frame_display_time_blocks)) = }, expected {sum(frames_in_stim_files) = }"

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
    # test_slow_processing_file()
    import pytest
    pytest.main(['-v', __file__, '--exitfirst'])