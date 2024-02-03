# npc_sync

Repackaging of `AllenSDK.SyncDataset` with extra functionality, compatible with data in the cloud.

[![PyPI](https://img.shields.io/pypi/v/npc_sync.svg?label=PyPI&color=blue)](https://pypi.org/project/npc_sync/)
[![Python version](https://img.shields.io/pypi/pyversions/npc_sync)](https://pypi.org/project/npc_sync/)

[![Coverage](https://img.shields.io/codecov/c/github/AllenInstitute/npc_sync?logo=codecov)](https://app.codecov.io/github/AllenInstitute/npc_sync)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/AllenInstitute/npc_sync/publish.yml?label=CI/CD&logo=github)](https://github.com/AllenInstitute/npc_sync/actions/workflows/publish.yml)
[![GitHub issues](https://img.shields.io/github/issues/AllenInstitute/npc_sync?logo=github)](https://github.com/AllenInstitute/npc_sync/issues)

# Usage
```bash
conda create -n npc_sync python>=3.9
conda activate npc_sync
pip install npc_sync
```
## Python
```python
>>> import npc_sync

>>> d = npc_sync.SyncDataset('s3://aind-ephys-data/ecephys_676909_2023-12-14_12-43-11/behavior_videos/20231214T124311.h5')
>>> d.line_labels[0]
'barcode_ephys'
>>> d.validate(opto=True, audio=True)

# note: some methods from the original `SyncDataset` are now property getters, including `line_labels`
```

# Development
See instructions in https://github.com/AllenInstitute/npc_sync/CONTRIBUTING.md and the original template: https://github.com/AllenInstitute/copier-pdm-npc/blob/main/README.md