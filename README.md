# Projection onto Total Variation ball

This repository contains an implementation of the Total Variation (TV) denoising algorithm as described in [Total Variation Projection with First Order schemes](https://hal.archives-ouvertes.fr/hal-00380491v3/document), by Fadili and Peyre.

Using the PyTorch framework, I basically re-wrote their algorithm to perform in batches, by using convolution matrices to act as finite difference operators. Currently, this only works for the case of 1-channel images (black and white), but the extension to color images should be straight forward in this setting.

### Citation
If you find this to be helpful in your research, please both myself and the original work:
```
@code{BatchwiseProjection,
  title={Batchwise Projection onto total variation norm ball},
  author={Pooladian, Aram-Alexandre},
  journal={Github},
  year={2019},
}

@article{fadili:hal-00380491,
  TITLE = {{Total Variation Projection with First Order Schemes}},
  AUTHOR = {Fadili, Jalal M. and Peyr{\'e}, Gabriel},
  URL = {https://hal.archives-ouvertes.fr/hal-00380491},
  JOURNAL = {{IEEE Transactions on Image Processing}},
  PUBLISHER = {{Institute of Electrical and Electronics Engineers}},
  VOLUME = {20},
  NUMBER = {3},
  PAGES = {657-669},
  YEAR = {2011},
  MONTH = Jan,
  PDF = {https://hal.archives-ouvertes.fr/hal-00380491/file/FadiliPeyreTVProj_twocolumns.pdf},
  HAL_ID = {hal-00380491},
  HAL_VERSION = {v3},
}
```
