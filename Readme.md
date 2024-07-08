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
2. To train the ReactAIvate model use 'ReactAIvate.ipynb' file.
3. For CRM generation, use 'CRM_Generation_using_ReactAIvate.ipynb' python file.
