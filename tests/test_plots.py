import npc_sync


def test_plots():
    dset = npc_sync.SyncDataset('s3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/20230821T124345.h5')
    dset.plot_stim_onsets().savefig('tests/output/test_stim_starts.png')
    dset.plot_stim_offsets().savefig('tests/output/test_stim_ends.png')

if __name__ == '__main__':
    import pytest
    
    pytest.main(['-s', __file__])
