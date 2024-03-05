import numpy as np
import pandas as pd
import torch

import selfies as sf
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import ExactMolWt

def get_selfie_and_smiles_encodings_for_dataset(file_path):
    
    """

    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.

    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string

    """

    df = pd.read_csv(file_path)
    
    smiles_list = np.asanyarray(df.smiles)

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    selfies_alphabet = list(all_selfies_symbols)

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len

### Get properties Using RDkit

def calculate_properties(smiles):
    m = MolFromSmiles(smiles)
    if m is None : return None
    MW = ExactMolWt(m)
    return [MW]

def stack_properties(smi_list):
    property_set = []

    for i in smi_list:
        properties = calculate_properties(i)
        property_set.append(properties)
    property_set = np.vstack(property_set)
    return torch.tensor(property_set)


# selfies_list, selfies_alphabet, largest_selfies_len, _,_,_ = get_selfie_and_smiles_encodings_for_dataset('datasets/0SelectedSMILES_QM9.txt')

# print(selfies_alphabet)
# print(selfies_list[0])
# print(selfies_alphabet.index('[nop]'))