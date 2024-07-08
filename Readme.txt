1. Create a conda environment using the following command
>> conda env create -f environment.yml

Activate the environment 
>> conda activate ReactAIvate

Install necessary libraries

>> conda install -c dglteam/label/cu113 dgl # Make sure to match the CUDA version with your system
>> pip install dgllife
>> pip install rdkit
>> pip install scikit-learn


2. To train the ReactAIvate model use 'ReactAIvate.ipynb' file.
3. For CRM generation, use 'CRM_Generation_using_ReactAIvate.ipynb' python file.
 