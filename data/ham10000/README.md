# HAM10000 — original release

Source: Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large collection of multi-source
dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018).
Harvard Dataverse doi:10.7910/DVN/DBW86T — licence CC BY-NC 4.0.

`HAM10000_metadata.csv` (690 KB, committed) is the only file needed to build and fingerprint the
canonical split. The two image zips (2.7 GB) are git-ignored; download them next to it on the machine
that trains or evaluates:

```bash
cd data/ham10000
curl -L -o HAM10000_images_part_1.zip "https://dataverse.harvard.edu/api/access/datafile/3172585"   # 1.37 GB, 5000 JPEGs
curl -L -o HAM10000_images_part_2.zip "https://dataverse.harvard.edu/api/access/datafile/3172584"   # 1.40 GB, 5015 JPEGs
```

(`curl -L -o HAM10000_metadata.csv "https://dataverse.harvard.edu/api/access/datafile/4338392?format=original"`
re-fetches the metadata.) `data.py` reads the JPEGs straight out of the zips — no extraction needed.
Extracted `*.jpg` files under this folder work too.

Check everything is in place:

```bash
python data.py --check-images     # 10015 images, published class counts, fingerprint 4b4cc59260945104
```
