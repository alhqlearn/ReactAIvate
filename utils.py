import pandas as pd
import random
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer

import torch
import os
import random
import numpy as np
import ast

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, display
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from rdkit import RDLogger 
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*') # switch off RDKit warning messages

def extra_non_reactive_class(dff):
    all_labels_app = []
    for i in range(dff.shape[0]):
        string_list = dff['smiles'][i]
        spp =string_list.split(".")
        random.shuffle(spp)
        spp = spp.pop(0)
        #spp = ('.').join(spp)
        all_labels = [spp,[],7]
        all_labels_app.append(all_labels)
    all_labels_app = pd.DataFrame(all_labels_app, columns = dff.columns)
    return all_labels_app
    

# Function to retrieve values at specified positions
def get_values_at_positions(my_tuple, positions_list):
    return [my_tuple[pos] for pos in positions_list if pos < len(my_tuple)]
    
def atom_finder(smiles, ids):
    mol = Chem.MolFromSmiles(smiles)
    if len(ids) == 0:
        shuffled_smiles = Chem.MolToSmiles(mol, doRandom=True)
        return shuffled_smiles, ids
    else: 
        atoms_interested = ast.literal_eval(ids)
        shuffled_smiles = Chem.MolToSmiles(mol, doRandom=True) #'CC(C)=O.CC(C)(C)c1ccccc1' #here we just have to give shuffled or randomized smiles
        shuffled_mol = Chem.MolFromSmiles(shuffled_smiles)
        shuffled_ids  = shuffled_mol.GetSubstructMatch(mol)
        new_ids = get_values_at_positions(shuffled_ids, atoms_interested)
        return shuffled_smiles, new_ids

def smiles_augmentation(df, num_aug, augmentation = True): 
    if augmentation == True:
      information = []
      for i in range(df.shape[0]):
          for _ in range(num_aug):
              smiles, ids = atom_finder(df['smiles'][i], df['atom_mapped'][i])
              react_random_list = [smiles, ids, df['class_label'][i]]
              information.append(react_random_list)
    else:
      information = []
      for i in range(df.shape[0]):
            react_random_list = [df['smiles'][i], ast.literal_eval(df['atom_mapped'][i]), df['class_label'][i]]
            information.append(react_random_list)    
    
    return information
        
def concat_feature_reactive_atom(graph_feat, changed_atoms):
    smiles_list = []
    target_list = []
    for i in range(len(changed_atoms)):
        node_lebel_tensor = torch.zeros(graph_feat[i].ndata['hv'].shape[0])
        #node_lebel_tensor[ast.literal_eval(changed_atoms[i][1])] = 1
        node_lebel_tensor[changed_atoms[i][1]] = 1
        node_lebel_tensor = node_lebel_tensor.unsqueeze(1)
        smiles_list.append(changed_atoms[i][0])
        target_list.append(changed_atoms[i][2])
        

        graph_feat[i].ndata['hv'] = torch.cat([graph_feat[i].ndata['hv'], node_lebel_tensor], dim=1)
    # Convert the list of numbers to a list of tensors
    target_tensor_list = [torch.tensor([x], dtype=torch.float32) for x in target_list]
    return list(zip(smiles_list, graph_feat, target_tensor_list))
    
def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))


    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, labels
    
    
def Canon_SMILES_similarity(smiles_list):
    # Convert all SMILES strings to molecular objects
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # Create an array of canonical smiles strings
    canonical_smiles_array = np.array([Chem.MolToSmiles(mol) if mol else None for mol in mol_list])

    # Use broadcasting to compare canonical smiles strings
    matrix = (canonical_smiles_array[:, None] != canonical_smiles_array).astype(np.float32)

    return torch.tensor(matrix)
    
    
