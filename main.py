import os
import sys
import torch
import yaml
from data_loader import \
    multiple_selfies_to_hot, multiple_smile_to_hot

from utils import *
from encoder_decoder import *
from train_cvae import *

### Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists("settings.yml"):
        settings = yaml.safe_load(open("settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return

    print('--> Acquiring data...')
    type_of_encoding = settings['data']['type_of_encoding']
    file_name_smiles = settings['data']['smiles_file']

    print('Finished acquiring data.')

    if type_of_encoding == 0:
        print('Representation: SMILES')
        _, _, _, encoding_list, encoding_alphabet, largest_molecule_len = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
        
        ############################################
        condition = stack_properties(encoding_list)
        ############################################

        print('--> Creating one-hot encoding...')
        data = multiple_smile_to_hot(encoding_list, largest_molecule_len,
                                     encoding_alphabet)
        print('Finished creating one-hot encoding.')

    elif type_of_encoding == 1:
        print('Representation: SELFIES')
        encoding_list, encoding_alphabet, largest_molecule_len, smi_list, _, _ = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
        
        ############################################
        condition = stack_properties(smi_list)
        padding_idx = encoding_alphabet.index('[nop]')
        ############################################

        print('--> Creating one-hot encoding...')
        data = multiple_selfies_to_hot(encoding_list, largest_molecule_len,
                                       encoding_alphabet)
        print('Finished creating one-hot encoding.')

    else:
        print("type_of_encoding not in {0, 1}.")
        return

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_mol_one_hot = len_max_molec * len_alphabet

    print(' ')
    print(f"Alphabet has {len_alphabet} letters, "
          f"largest molecule is {len_max_molec} letters.")

    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']

    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']

    vae_encoder = VAEEncoder(vocab_size=len_alphabet,padding_idx=padding_idx,in_dimension=len_max_mol_one_hot,
                             **encoder_parameter).to(device)
    vae_decoder = VAEDecoder(**decoder_parameter,
                             out_dimension=len(encoding_alphabet)).to(device)

    print('*' * 15, ': -->', device)

    data = torch.tensor(data, dtype=torch.float).to(device)

    train_valid_test_size = [0.5, 0.5, 0.0]
    indices = torch.randperm(data.size()[0]) ###########
    data = data[indices]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data_train = data[0:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]


    ##############################################
    condition = condition[indices] ############33
    condition_train = condition[0:idx_train_val]
    condition_valid = condition[idx_train_val:idx_val_test]
    ##############################################

    print("start training")
    train_model(**training_parameters,
                vae_encoder=vae_encoder,
                vae_decoder=vae_decoder,
                batch_size=batch_size,
                data_train=data_train,
                data_valid=data_valid,
                alphabet=encoding_alphabet,
                type_of_encoding=type_of_encoding,
                sample_len=len_max_molec,
                condition_train = condition_train,
                condition_valid = condition_valid)


### Run

if __name__ == '__main__':
    try:
        main()
        writer.close()
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)

### Check
def load_models():
    out_dir = './saved_models'
    encoder = torch.load('{}/E_selfies_2'.format(out_dir))  # Load saved encoder
    decoder = torch.load('{}/D_selfies_2'.format(out_dir))  # Load saved decoder
    return encoder, decoder

# 저장된 모델 불러오기
# encoder, decoder = load_models()
# a,b,c,d,e,f = get_selfie_and_smiles_encodings_for_dataset('datasets/0SelectedSMILES_QM9.txt')
# calculate_properties('CCCCCCCCCCCC')
# a = latent_space_quality_test(encoder,decoder,b,500,c,torch.tensor([100]))
# pro_set = 0
# for i in a:
#     smi = sf.decoder(i)
#     # print(smi)
#     pro_set += int(calculate_properties(smi)[-1])
#     # print(is_correct_smiles(i))

# print(pro_set/500)
    
# ### Trash
# torch.manual_seed(42)

# drug = torch.tensor([[1],[2],[3],[4],[5]])
# target = torch.tensor([[1],[2],[3],[4],[5]])

# # 동일한 무작위 인덱스 생성
# indices = torch.randperm(drug.size()[0])

# # 데이터 섞기
# drug = drug[indices]
# target = target[indices]


# print(drug)

# print(target)
