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


def test_moved_names_are_deprecated_aliases_of_npc_stim() -> None:
    import warnings

    import pytest

    npc_stim = pytest.importorskip("npc_stim")
    if not hasattr(npc_stim, "display_times"):
        # bootstrapping the release: npc_stim with display_times needs this npc_sync on PyPI first
        pytest.skip("installed npc_stim predates the move of display times")

    for name in ("FRAME_RATE", "CONSTANT_MONITOR_LAG", "get_frame_display_times", "asym_round"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert getattr(npc_sync, name) is getattr(npc_stim.display_times, name)
            assert getattr(npc_sync.sync, name) is getattr(npc_stim.display_times, name)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_unknown_attribute_raises() -> None:
    import pytest

    with pytest.raises(AttributeError):
        _ = npc_sync.not_a_real_name
