# UVCGAN2

## First time

From the repository root:

```powershell
pip install torch torchvision
pip install -r requirements.txt
python uvcgan_training/setup_uvcgan.py
```

Set `BASE_DIR` in `data_processing/config.py`.

```powershell
python process_data.py

cd uvcgan_training
$env:UVCGAN2_DATA   = "..."
$env:UVCGAN2_OUTDIR = "..."
python pretrain_crystals.py
python train_crystals.py
```

## Later times

From the repository root:

```powershell
python uvcgan_training/setup_uvcgan.py
python process_data.py

cd uvcgan_training
$env:UVCGAN2_DATA   = "..."
$env:UVCGAN2_OUTDIR = "..."
python pretrain_crystals.py
python train_crystals.py
```

Skip `setup_uvcgan.py` if already installed. Skip `process_data.py` if the dataset is already built.
