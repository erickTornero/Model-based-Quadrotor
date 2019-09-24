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

        x_train, x_test, y_train, y_test  =   train_test_split(X_data, target, test_size=self.split_ratio, random_state=42, shuffle=True)


        n_batches   =   y_train.shape[0]//self.batch_size# + (1 if target.shape[0] % self.batch_size > 0 else 0)

        n_batches_test  = y_test.shape[0]//self.batch_size if y_test.shape[0] >= self.batch_size else 1 
        # TODO: Last N<batch_size - 1 data in the epoch could not be taking into account in training
        #
        
        for n_epoch in range(self.nepochs):
            """ Training Step """
            loss_per_epoch  =   []    
            self.index  =   0
            for n_b in range(n_batches):
                x_batch =   x_train[self.index:self.index + self.batch_size,:]
                y_batch =   y_train[self.index:self.index + self.batch_size,:]
                self.index  +=  self.batch_size

                X_tensor    =   torch.tensor(x_batch, dtype=torch.float32, device=self.device)
                Y_tensor    =   torch.tensor(y_batch, dtype=torch.float32, device=self.device)

                Y_pred      =   self.network(X_tensor)

                """ Training steps """
                self.optimizer.zero_grad()
                output      =   torch.mean(torch.sum((Y_tensor - Y_pred)**2, axis=1))
                #output      =   self.loss(Y_pred, Y_tensor)
                loss_per_epoch.append(output.item())
                output.backward()
                self.optimizer.step()
            # Validation Loss
            """ Validation step """
            loss_per_val    =   []
            self.index      =   0
            for n_b in range(n_batches_test):
                x_batch_t =   x_test[self.index:self.index + self.batch_size,:]
                y_batch_t =   y_test[self.index:self.index + self.batch_size,:]
                X_tensor_test   =   torch.tensor(x_batch_t, dtype=torch.float32, device=self.device)
                Y_tensor_test   =   torch.tensor(y_batch_t, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    Y_pred_test     =   self.network(X_tensor_test)
                    output_test     =   torch.mean(torch.sum((Y_tensor_test - Y_pred_test)**2, axis=1))
                    loss_per_val.append(output_test.item())
            print('Loss epoch {} -> (train, test) loss-> ({:3.4f}, {:3.4f})'.format(n_epoch + 1, sum(loss_per_epoch)/len(loss_per_epoch), sum(loss_per_val)/len(loss_per_val)))
