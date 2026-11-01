# Data folder

The full dataset is **not** stored in Git (it is ~47 MB and has its own terms
of use). Download it yourself and place the two CSV files here:

```
data/
├── raw/
│   ├── UNSW_NB15_training-set.csv   <- 175,341 records (used for training)
│   └── UNSW_NB15_testing-set.csv    <- 82,332 records (used for final testing)
└── demo/
    └── demo_traffic.csv             <- 500-row excerpt, included for the dashboard
```

## Where to download

Official source (UNSW Canberra Cyber Range Lab):
**https://research.unsw.edu.au/projects/unsw-nb15-dataset**

1. Open the page and follow the dataset download link (it opens a shared folder).
2. Open the folder **"Training and Testing Sets"**.
3. Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.
4. Copy both files into `data/raw/` with exactly these names.

The same two files are also mirrored on Kaggle and GitHub, but prefer the
official source. The loader validates the columns, so a file with a different
schema is rejected with a clear message rather than silently used.

Check the files are in place and valid:

```bash
python -c "from src.data_loader import load_labeled_dataset as L; from src import config as c; print(len(L(c.TRAIN_FILE)), len(L(c.TEST_FILE)))"
```

It should print `175341 82332`.

## Notes on the official files

- The training file begins with an invisible UTF-8 byte-order mark (BOM) before
  the `id` column name. `src/data_loader.py` strips it automatically.
- Despite the names, the training file is the larger one (175,341 rows).
- `label` is the binary target (0 = normal, 1 = attack).
- `attack_cat` is the attack family (`Normal`, `Generic`, `Exploits`, ...). It is
  derived from the target, so it is **never** used as a model input.
- `id` is a row counter and is also never used as an input.

## The demo file

`data/demo/demo_traffic.csv` is a 500-row excerpt of the official testing set
(every attack category represented), so the dashboard works without uploading
anything. Re-create it after downloading the dataset with:

```bash
python -m src.make_demo
```

## Citation

If you use UNSW-NB15, cite:

N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network
intrusion detection systems (UNSW-NB15 network data set)," *2015 Military
Communications and Information Systems Conference (MilCIS)*, 2015.
