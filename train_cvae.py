import os
import time
import numpy as np
import torch
from torch import nn
import selfies as sf
from torch.utils.tensorboard import SummaryWriter
from model_utils import *

current_mode = 'SELFIES'
try:
    os.mkdir('./molecular_generator/selfies_cvae_result')
    saved_loc = os.path.join('./molecular_generator/selfies_cvae_result',current_mode)
    os.mkdir(saved_loc)
    print(f'Save file loc : {saved_loc}')
    writer = SummaryWriter(saved_loc)
except:
    saved_loc = os.path.join('./molecular_generator/selfies_cvae_result',current_mode)
    print(f'Save file loc : {saved_loc}')
    writer = SummaryWriter(saved_loc)

def save_models(encoder, decoder):
    out_dir = './saved_models'
    torch.save(encoder, '{}/E_selfies'.format(out_dir))
    torch.save(decoder, '{}/D_selfies'.format(out_dir))

### Train
def train_model(vae_encoder, vae_decoder,
                data_train, data_valid, num_epochs, batch_size,
                lr_enc, lr_dec, KLD_alpha,
                sample_num, sample_len, alphabet, type_of_encoding,
                condition_train, condition_valid):
    """
    Train the Variational Auto-Encoder
    """

    print('num_epochs: ', num_epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    data_train = data_train.clone().detach().to(device)
    #################################################################
    condition_train = condition_train.clone().detach().to(device)
    #################################################################
    num_batches_train = int(len(data_train) / batch_size)

    quality_valid_list = [0, 0, 0, 0]
    for epoch in range(num_epochs):

        indices = torch.randperm(data_train.size()[0]) ################

        data_train = data_train[indices]
        ############################################################################
        condition_train = condition_train[indices]
        ############################################################################

        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator

            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx: stop_idx] # [128, 21, 18] [batch, largest_molecule size, vocab_size]
            batch_c = condition_train[start_idx: stop_idx].to(torch.float) # [128,1] [batch, condition]

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot,batch_c)

            # initialization hidden internal state of RNN (RNN has two inputs
            # and two outputs:)
            #    input: latent space & hidden state
            #    output: one-hot encoding of one character of molecule & hidden
            #    state the hidden state acts as the internal memory
            latent_points = latent_points.unsqueeze(0)
            hidden = vae_decoder.init_hidden(batch_size=batch_size)

            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot = torch.zeros_like(batch, device=device)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points, batch_c, hidden) #################
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

            # compute ELBO
            CE, KLD, KLD_alpha = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha)
            loss = CE + KLD_alpha*KLD

        

            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 60 == 0:
                end = time.time()

                # assess reconstruction quality
                quality_train = compute_recon_quality(batch, out_one_hot)
                quality_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                                     data_valid, condition_valid, batch_size)

                report = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| ' \
                         'quality: %.4f | quality_valid: %.4f)\t' \
                         'ELAPSED TIME: %.5f' \
                         % (epoch, batch_iteration, num_batches_train,
                            loss.item(), quality_train, quality_valid,
                            end - start)
                print(report)
                start = time.time()
        ###################################################  
        writer.add_scalar('./Train/loss',loss, epoch) # add loss value
        ###################################################

        quality_valid = quality_in_valid_set(vae_encoder, vae_decoder,
                                             data_valid, condition_valid, batch_size) 
        # np.mean(set of compute_recon_qulity)
        quality_valid_list.append(quality_valid)

        # only measure validity of reconstruction improved
        quality_increase = len(quality_valid_list) \
                           - np.argmax(quality_valid_list)
        
        # Just watch valid score
        corr_test, unique_test = latent_space_quality(vae_encoder, vae_decoder,
                                                type_of_encoding, alphabet,
                                                sample_num, sample_len)

        writer.add_scalar('./Test/Validity',corr_test * 100. / sample_num, epoch)
        writer.add_scalar('./Test/Diversity',unique_test * 100. / sample_num, epoch)
        writer.add_scalar('./Test/Reconstruction',quality_valid, epoch) 
        #######################

        if quality_increase == 1 and quality_valid_list[-1] > 50.:
            corr, unique = latent_space_quality(vae_encoder, vae_decoder,
                                                type_of_encoding, alphabet,
                                                sample_num, sample_len)
            
            save_models(vae_encoder, vae_decoder)
        else:
            corr, unique = -1., -1.

        report = 'Validity: %.5f %% | Diversity: %.5f %% | ' \
                 'Reconstruction: %.5f %%' \
                 % (corr * 100. / sample_num, unique * 100. / sample_num,
                    quality_valid)
        print(report)

        #################################################################################################################
        if type_of_encoding == 0 and epoch % 50 == 0:
            molecule_set = latent_space_quality_test(vae_encoder,vae_decoder,alphabet,10,sample_len,torch.tensor([122]))
            molecule_str = ', '.join(molecule_set)
            writer.add_text('./Test/Sample',molecule_str,epoch)

        elif type_of_encoding == 1 and epoch % 50 == 0:
            molecule_set = latent_space_quality_test(vae_encoder,vae_decoder,alphabet,10,sample_len,torch.tensor([122]))
            smi_set = []
            for i in molecule_set:
                try:
                    smi = sf.decoder(i)
                    smi_set.append(smi)
                except:
                    pass
            molecule_str = ', '.join(smi_set)
            writer.add_text('./Test/Sample',molecule_str,epoch)
        #################################################################################################################


        if quality_valid_list[-1] < 70. and epoch > 200:
            break

        if quality_increase > 20: # Early stop 
            print('Early stopping criteria')
            break

