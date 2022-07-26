import torch
import torch.nn as nn

class Autoencoder_deep_layer(nn.Module):
    def __init__(self,
                input_size,
                activation,
                noise=None
                ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size1 = (input_size // 5)*4
        self.hidden_size2 = (input_size // 5)*3
        self.hidden_size3 = (input_size // 5)*2
        self.hidden_size4 = (input_size // 5)*1
        self.activation = activation
        self.noise = noise

        self.encoder = nn.Sequential(

            # size(self.input_size) -> size(hidden_size1)
            nn.Linear(input_size, self.hidden_size1),
            self.activation,

            # size(hidden_size1) -> size(hidden_size2)
            nn.Linear(self.hidden_size1, self.hidden_size2),
            self.activation,

            # size(hidden_size2) -> size(hidden_size3)
            nn.Linear(self.hidden_size2, self.hidden_size3),
            self.activation,

            # size(hidden_size3) -> size(hidden_size4)
            nn.Linear(self.hidden_size3, self.hidden_size4),
        )

        self.decoder = nn.Sequential(

            # size(hidden_size4) -> size(hidden_size3)
            nn.Linear(self.hidden_size4, self.hidden_size3),
            self.activation,

            # size(hidden_size3) -> size(hidden_size2)
            nn.Linear(self.hidden_size3, self.hidden_size2),
            self.activation,

            # size(hidden_size2) -> size(hidden_size1)
            nn.Linear(self.hidden_size2, self.hidden_size1),
            self.activation,

            # size(hidden_size1) -> size(sefl.input_size)
            nn.Linear(self.hidden_size1, input_size)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def Activation_choose(activation):

    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ReakyLELU':
        return nn.LeakyReLU()
    else:
        pass

def Model_selector(size,
                    model_name,
                    activation,
                    noise):

    if model_name == 'Autoencoder_base':

        # Will be deprecated
        activation = Activation_choose(activation)
        model = Autoencoder_base(44,30,20,10,activation)

    elif model_name == 'Autoencoder_deep_layer':

        activation = Activation_choose(activation)
        model = Autoencoder_deep_layer(size,activation,noise)

    else:
        # No model
        return False

    return model
