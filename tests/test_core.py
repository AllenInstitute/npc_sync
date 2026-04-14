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
