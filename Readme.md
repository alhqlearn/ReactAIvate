## Overview
The work introduces a new dataset and related task of predicting single reaction steps which is required to predict chemical reaction mechanisms. A model is introduced that simultaneously predicts reaction steps and reactive atoms, using an attention based graph neural network based architecture.
### Environmental Setup

```
conda env create -f environment.yml
conda activate ReactAIvate
conda install -c dglteam/label/cu113 dgl # Make sure to match the CUDA version with your system
pip install dgllife
pip install rdkit
pip install scikit-learn
```
### Training 
To train the ReactAIvate model use 'ReactAIvate.ipynb' file.

For CRM generation, use 'CRM_Generation_using_ReactAIvate.ipynb' python file.

###  Citations

If you find this code or work useful in your research, please cite:

**Hoque, A.; Das, M.; Baranwal, M.; Sunoj, R. B.** *ReactAIvate: A Deep Learning Approach to Predicting Reaction Mechanisms and Unmasking Reactivity Hotspots.* *Proceedings in Artificial Intelligence (ECAI 2024),* Volume 392, 2024, Pages 2645–2652. DOI: [10.3233/FAIA240796](https://doi.org/10.3233/FAIA240796)

###  BibTeX
```bibtex
@inproceedings{Hoque2024_ReactAIvate,
  author    = {Hoque, A. and Das, M. and Baranwal, M. and Sunoj, R. B.},
  title     = {ReactAIvate: A Deep Learning Approach to Predicting Reaction Mechanisms and Unmasking Reactivity Hotspots},
  booktitle = {Proceedings in Artificial Intelligence (ECAI 2024)},
  year      = {2024},
  volume    = {392},
  pages     = {2645--2652},
  doi       = {10.3233/FAIA240796}
}

