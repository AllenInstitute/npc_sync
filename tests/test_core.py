import pathlib

import h5py

import npc_sync


def test_import_package():
    pass


def test_no_stim_paths_and_meta_data() -> None:
    """Constructing SyncDataset without stim_paths leaves stim_paths as None,
    and meta_data is accessible and contains expected keys."""
    s = npc_sync.SyncDataset(
        "s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5"
    )
    assert s.stim_paths is None
    meta = s.meta_data
    assert isinstance(meta, dict)
    assert "line_labels" in meta


def test_init_from_instance_and_meta_data() -> None:
    """Constructing SyncDataset from an existing instance reuses it,
    and meta_data is still accessible with expected keys."""
    s = npc_sync.SyncDataset(
        "s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5"
    )
    s2 = npc_sync.SyncDataset(s)
    assert s2.stim_paths is None
    meta = s2.meta_data
    assert isinstance(meta, dict)
    assert "line_labels" in meta


def test_diode_box_frame_interval_is_read_from_stim_file(
    tmp_path: pathlib.Path,
) -> None:
    current_stim = tmp_path / "current.hdf5"
    with h5py.File(current_stim, "w") as stim_data:
        stim_data["diodeBoxFrameInterval"] = 3

    old_stim = tmp_path / "old.hdf5"
    with h5py.File(old_stim, "w"):
        pass

    sync = object.__new__(npc_sync.SyncDataset)
    sync.block_index_to_stim_path = {
        0: current_stim,
        1: old_stim,
        2: None,
    }

    assert sync._get_diode_box_frame_interval(0) == 3
    assert sync._get_diode_box_frame_interval(1) == 1
    assert sync._get_diode_box_frame_interval(2) == 1
