import numpy as np
import torch


class PassiveEncLayer:
    def __init__(self, in_nodes, out_nodes, eta, cryptosystem, messenger):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.eta = eta
        self.cryptosystem = cryptosystem
        self.messenger = messenger

        self.w_acc = np.random.rand(in_nodes, out_nodes).astype(np.float32)

    def fed_forward(self, a):
        enc_a = self.cryptosystem.encrypt_vector(torch.flatten(a)) # return a Python list
        enc_a = np.array(enc_a).reshape(a.shape) # convert it to numpy array
        self.messenger.send(enc_a)
        enc_z_tilde = self.messenger.recv() # numpy array, shape: (bs, output_nodes)
        z_tilde = self.cryptosystem.decrypt_vector(enc_z_tilde.flatten())
        z_tilde = torch.tensor(z_tilde).reshape(enc_z_tilde.shape)
        # z_tilde = np.array(z_tilde).reshape(enc_z_tilde.shape)
        z_clear = z_tilde - torch.matmul(a, torch.from_numpy(self.w_acc))
        return z_clear # tensor, shape: (bs, output_nodes)

    def fed_backward(self):
        enc_w_tilde_grad = self.messenger.recv() # numpy array, shape: (input_nodes, output_nodes)
        w_tilde_grad = self.cryptosystem.decrypt_vector(enc_w_tilde_grad.flatten())
        w_tilde_grad = np.array(w_tilde_grad).reshape(enc_w_tilde_grad.shape)

        w_curr = np.random.rand(self.in_nodes, self.out_nodes).astype(np.float32)
        w_tilde_grad = w_tilde_grad - w_curr / self.eta

        enc_w_acc = self.cryptosystem.encrypt_vector(self.w_acc.flatten())
        enc_w_acc = np.array(enc_w_acc).reshape(self.w_acc.shape)
        self.messenger.send([w_tilde_grad, enc_w_acc])

        # update w_acc
        self.w_acc = self.w_acc + w_curr


class ActiveEncLayer:
    def __init__(self, in_nodes, out_nodes, eta, cryptosystem, messenger):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.eta = eta
        self.cryptosystem = cryptosystem
        self.messenger = messenger

        self.w_tilde = np.random.rand(in_nodes, out_nodes).astype(np.float32)
        self.curr_enc_a = None

    def fed_forward(self):
        enc_a = self.messenger.recv() # is a numpy array, shape: (bs, input_nodes)
        self.curr_enc_a = enc_a
        # enc_z_tilde = np.matmul(enc_a, self.w_tilde.numpy())
        enc_z_tilde = np.matmul(enc_a, self.w_tilde) # shape: (bs, output_nodes)
        self.messenger.send(enc_z_tilde)

    def fed_backward(self, grad):
        grad = grad.numpy()
        enc_w_tilde_grad = np.matmul(self.curr_enc_a.transpose(), grad)
        self.messenger.send(enc_w_tilde_grad)

        w_tilde_grad, enc_w_acc = self.messenger.recv()

        enc_a_grad = np.matmul(grad, self.w_tilde.transpose()) \
                     - np.matmul(grad, enc_w_acc.transpose())
        self.w_tilde = self.w_tilde - self.eta * w_tilde_grad

        return enc_a_grad