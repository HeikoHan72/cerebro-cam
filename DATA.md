# Data

This repository does **not** contain the datasets. `scripts/00_setup_data.py` sets them up from the
original Kaggle downloads. It identifies each archive by its content, compares its SHA256 checksum with
the version used here and checks the image counts.

## Datasets used

### 1. Brain Tumor MRI Dataset (training and internal test)

| | |
|---|---|
| Author | Masoud Nickparvar |
| Source | <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset> |
| DOI | [10.34740/kaggle/dsv/14832123](https://doi.org/10.34740/kaggle/dsv/14832123) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Images | Training 5,600 (1,400 per class), Testing 1,600 (400 per class) |
| Classes | glioma, meningioma, notumor, pituitary |
| SHA256 of the archive used | `882817250048c78ef7a759cf23e540d7b581f2327b16663c9d3db12f5d2ffdb4` |

According to the dataset description, it combines three public datasets:

- figshare (Cheng et al., see below)
- SARTAJ: its glioma images were removed because their labels were inconsistent, and replaced by
  figshare images
- Br35H: the source of all no-tumor images

Version 2 states that duplicates were removed and the train/test overlap was eliminated. The duplicate
audit in this repository still finds near-identical images (see README, "Duplicate audit").

### 2. Brain Tumor Classification (MRI) (external test)

| | |
|---|---|
| Authors | Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge, Swati Kanchan |
| Source | <https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri> (repository: <https://github.com/sartajbhuvaji/brain-tumor-classification-dataset>) |
| DOI | [10.34740/kaggle/dsv/12745533](https://doi.org/10.34740/kaggle/dsv/12745533) |
| License | MIT |
| Images | 3,264 (Training 2,870, Testing 394) |
| Classes | glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor (mapped to the four classes above) |
| SHA256 of the archive used | `b97a99ee09daf073bfa3e49a5a09002a582b4ee8478a26d3af3a2c67be62c79c` |

This dataset is part of dataset 1, so it is **not an independent test set**. All 3,264 images are
used as one external pool. `results/duplicates/external_overlap.csv` shows, for every image, whether it
also occurs in the training data.

## Upstream sources of dataset 1

| Dataset | Source | License |
|---|---|---|
| figshare brain tumor dataset (Jun Cheng): 3,064 T1-weighted contrast-enhanced images from 233 patients | <https://doi.org/10.6084/m9.figshare.1512427> | CC BY 4.0 |
| Br35H: Brain Tumor Detection 2020 (Ahmed Hamada) | <https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection> | Stated on Kaggle as "Data files © Original Authors" |
| SARTAJ | see dataset 2 | MIT |

Note: Dataset 1 is published under CC BY 4.0. Its upstream source Br35H, however, gives no open license
on its Kaggle page. The figures in `results/figures/` contain a few example images from dataset 1,
including no-tumor images, for illustration only.

## References

If you use the data, please cite the original sources:

- Nickparvar, M. *Brain Tumor MRI Dataset*. Kaggle. https://doi.org/10.34740/kaggle/dsv/14832123
- Bhuvaji, S., Kadam, A., Bhumkar, P., Dedge, S., Kanchan, S. *Brain Tumor Classification (MRI)*.
  Kaggle. https://doi.org/10.34740/kaggle/dsv/12745533
- Cheng, J. *brain tumor dataset*. figshare. https://doi.org/10.6084/m9.figshare.1512427
- Cheng, J., et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation
  and Partition." *PLoS ONE* 10(10), 2015.
- Cheng, J., et al. "Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector
  Representation." *PLoS ONE* 11(6), 2016.
- Hamada, A. *Br35H: Brain Tumor Detection 2020*. Kaggle.
  https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

## Folder structure after setup

```
data/
├── raw/                 the two downloaded archives (any file name)
├── nickparvar/          Training/<class>/, Testing/<class>/
├── sartaj/              Training/<class>_tumor/, Testing/<class>_tumor/
├── processed/           150x150 JPEGs created by 03_preprocessing.py
└── external/            blinded external images and mapping (06_prepare_external_data.py)
```
