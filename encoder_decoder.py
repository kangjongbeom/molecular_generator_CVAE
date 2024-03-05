import torch
from torch import nn


### Encoder / Decoder ###
class VAEEncoder(nn.Module):

    def __init__(self,vocab_size,padding_idx,in_dimension, layer_1d, layer_2d, layer_3d,
                 latent_dimension, condition_dimension = 1):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(VAEEncoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.embedding = nn.Embedding(vocab_size, vocab_size*5,padding_idx=padding_idx) ## padding_idx ?

        self.encode_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=2,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
        )

        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension+condition_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_2d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_2d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, c):
        """
        Pass throught the Encoder
        """
        # Get results of encoder network
        c = c.to(x.device) 
        # x = torch.argmax(x,dim=1) # one-hot -> numerical encoding
        # x = self.embedding(x) # embedding / shape : [128,76]
        # x = torch.unsqueeze(x,dim=1) # [1, 128, 90]
        # x = x.permute(1,0,2) # [128, 1 , 76]
        # x = self.encode_cnn(x) # [128, 32, 9]

        # x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,c), dim=1) # concat

        h1 = self.encode_nn(x)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    

# class VAEDecoder(nn.Module):

#     def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,
#                  out_dimension, condition_dimension=1):
#         """
#         Through Decoder
#         """
#         super(VAEDecoder, self).__init__()
#         self.latent_dimension = latent_dimension + condition_dimension

#         # CNN Decoder
#         self.decode_cnn = nn.Sequential(
#             nn.ConvTranspose1d(in_channels=latent_dimension+condition_dimension, out_channels=32,kernel_size=3,stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose1d(in_channels=32,out_channels=16,kernel_size=3,stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose1d(in_channels=16,out_channels=8,kernel_size=3,stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose1d(in_channels=8,out_channels=1,kernel_size=3,stride=2)
#         )

#         self.decode_FC = nn.Sequential(
#             nn.Linear(31, out_dimension),
#         )

#     def forward(self, z,c):
#         """
#         A forward pass throught the entire model.
#         """
#         c_expanded = c.unsqueeze(0).expand(z.size(0),-1,-1) 
#         c_expanded = c_expanded.to(z.device)
#         z = torch.cat((z,c_expanded), dim=2) # shape : 1 128 51

#         z = z.permute(1,2,0) # 51 1 128

#         # Decode
#         l1 = self.decode_cnn(z) # 128 1 24
#         decoded = self.decode_FC(l1)  # fully connected layer

#         return decoded
    
class VAEDecoder(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,
                 out_dimension, condition_dimension=1):
        """
        Through Decoder
        """
        super(VAEDecoder, self).__init__()
        self.latent_dimension = latent_dimension + condition_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension+condition_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)

    def forward(self, z,c, hidden):
        """
        A forward pass throught the entire model.
        """
        c_expanded = c.unsqueeze(0).expand(z.size(0),-1,-1) ############ for match with z shape
        c_expanded = c_expanded.to(z.device)
        z = torch.cat((z,c_expanded), dim=2) 

        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden