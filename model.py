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



import torch.nn as nn
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.model import AttentiveFPGNN
from dgllife.model import AttentiveFPReadout



class AttentiveFPPredictor_rxn(nn.Module):

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 num_timesteps=1,
                 graph_feat_size=200,
                 n_tasks=8,
                 dropout=0.1):
        super(AttentiveFPPredictor_rxn, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, n_tasks)
        )
        self.node_predict = nn.Sequential(nn.Linear(graph_feat_size, 1), nn.Sigmoid()
        )
    def forward(self, g, node_feats, edge_feats, get_node_weight=False):

        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            node_feat_mod = self.node_predict(node_feats)
            return self.predict(g_feats), node_weights, node_feat_mod, g_feats
        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            node_feat_mod = self.node_predict(node_feats)
            return self.predict(g_feats), node_feat_mod, g_feats
            
            
            
def weighted_binary_cross_entropy(output, target, weights=None):        
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] *(target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    #print(loss)
    return torch.neg(torch.mean(loss))
    