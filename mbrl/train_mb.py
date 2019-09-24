from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self, network, batch_sz, nepochs, split_ratio, lr, device, optimizer=None, randn_seed=42):
        self.network        =   network
        self.device         =   device
        self.loss           =   nn.MSELoss()
        """ Training Process parameters """
        
        self.batch_size     =   batch_sz
        self.nepochs        =   nepochs
        self.split_ratio    =   min(max(split_ratio, 0.1), 0.4)
        self.lr             =   lr
        self.optimizer          =   optimizer if optimizer is not None else optim.Adam(self.network.parameters(), lr=lr)

        self.index          =   0


    def fit(self, X_data, target):
        """ Data must be compatible in shapes"""
        assert X_data.shape[0] ==target.shape[0]

        x_train, x_test, y_train, y_target  =   train_test_split(X_data, target, self.split_ratio, random_state=42, shuffle=True)


        n_batches   =   target.shape[0]/self.batch_size# + (1 if target.shape[0] % self.batch_size > 0 else 0)

        # TODO: Last N<batch_size - 1 data in the epoch could not be taking into account in training
        #
        for n_epoch in range(self.nepochs):
            
            for n_b in range(n_batches):
                x_batch =   x_train[self.index:self.batch_size,:]
                y_batch =   y_train[self.index:self.batch_size,:]
                self.index  +=  self.batch_size

                X_tensor    =   torch.tensor(x_batch, dtype=torch.float32, device=self.device)
                Y_tensor    =   torch.tensor(y_batch, dtype=torch.float32, device=self.device)

                Y_pred      =   self.network(X_tensor)

                """ Training steps """
                self.optimizer.zero_grad()
                output      =   self.loss(Y_pred, Y_tensor)

                output.backward()
                self.optimizer.step()



