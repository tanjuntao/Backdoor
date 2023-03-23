import multiprocessing
import os
from typing import Optional

import numpy as np
import torch

from linkefl.base import BaseCryptoSystem, BaseMessenger, BasePartialCryptoSystem
from linkefl.common.const import Const
from linkefl.crypto.paillier import cipher_matmul, encode


class PassiveEncLayer:
    def __init__(
        self,
        in_nodes: int,
        out_nodes: int,
        eta: float,
        cryptosystem: BaseCryptoSystem,
        messenger: BaseMessenger,
        num_workers: int = 1,
        random_state: Optional[int] = None,
        encode_precision: float = 0.001,
    ):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.eta = eta
        self.cryptosystem = cryptosystem
        self.messenger = messenger
        self.num_workers = num_workers
        if self.num_workers > 1:
            if cryptosystem.type == Const.PAILLIER:
                # using thread_pool for Paillier encrypt_vector
                self.enc_pool = multiprocessing.pool.ThreadPool(self.num_workers)
            else:
                # using process_pool for FastPaillier encrypt_vector
                self.enc_pool = multiprocessing.pool.Pool(self.num_workers)
        else:
            self.enc_pool = None
        if random_state is not None:
            np.random.seed(random_state)
        self.encode_precision = encode_precision

        self.w_acc: np.ndarray = np.random.rand(in_nodes, out_nodes).astype(np.float32)

    def fed_forward(self, a):
        a = a.to("cpu")
        enc_a = self.cryptosystem.encrypt_vector(
            torch.flatten(a), pool=self.enc_pool
        )  # return a Python list
        enc_a = np.array(enc_a).reshape(a.shape)  # convert it to numpy array
        self.messenger.send(enc_a)
        enc_z_tilde = self.messenger.recv()  # numpy array, shape: (bs, output_nodes)
        z_tilde = self.cryptosystem.decrypt_vector(enc_z_tilde.flatten())
        z_tilde = [float(val) for val in z_tilde]  # TODO: strange mpfr dtype
        z_tilde = torch.tensor(z_tilde).reshape(enc_z_tilde.shape)
        # z_tilde = np.array(z_tilde).reshape(enc_z_tilde.shape)
        z_clear = z_tilde - torch.matmul(a, torch.from_numpy(self.w_acc))
        return z_clear  # tensor, shape: (bs, output_nodes)

    def fed_backward(self):
        enc_w_tilde_grad = (
            self.messenger.recv()
        )  # numpy array, shape: (input_nodes, output_nodes)
        w_tilde_grad = self.cryptosystem.decrypt_vector(enc_w_tilde_grad.flatten())
        w_tilde_grad = [float(val) for val in w_tilde_grad]  # TODO: strange mpfr dtype
        w_tilde_grad = np.array(w_tilde_grad).reshape(enc_w_tilde_grad.shape)

        w_curr = np.random.rand(self.in_nodes, self.out_nodes).astype(np.float32)
        w_tilde_grad = w_tilde_grad - w_curr / self.eta

        enc_w_acc = self.cryptosystem.encrypt_vector(
            self.w_acc.flatten(), pool=self.enc_pool
        )
        enc_w_acc = np.array(enc_w_acc).reshape(self.w_acc.shape)
        self.messenger.send([w_tilde_grad, enc_w_acc])

        # update w_acc
        self.w_acc = self.w_acc + w_curr

    def close_pool(self):
        if self.enc_pool is not None:
            self.enc_pool.close()


class ActiveEncLayer:
    def __init__(
        self,
        in_nodes: int,
        out_nodes: int,
        eta: float,
        cryptosystem: BasePartialCryptoSystem,
        messenger: BaseMessenger,
        num_workers: int = 1,
        random_state: Optional[int] = None,
        encode_precision: float = 0.001,
    ):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.eta = eta
        self.cryptosystem = cryptosystem
        self.messenger = messenger
        self.num_workers = num_workers
        if self.num_workers > 1:
            self.thread_pool = multiprocessing.pool.ThreadPool(self.num_workers)
            self.scheduler_pool = multiprocessing.pool.ThreadPool(self.num_workers)
            self.encode_pool = multiprocessing.pool.Pool(self.num_workers)
        else:
            self.thread_pool = None
            self.scheduler_pool = None
            self.encode_pool = None
        if random_state is not None:
            np.random.seed(random_state)
        self.encode_precision = encode_precision

        self.w_tilde: np.ndarray = np.random.rand(in_nodes, out_nodes).astype(
            np.float32
        )
        self.curr_enc_a: Optional[np.ndarray] = None

    def fed_forward(self):
        enc_a = self.messenger.recv()  # is a numpy array, shape: (bs, input_nodes)
        self.curr_enc_a = enc_a
        w_tilde_encode = encode(
            raw_data=self.w_tilde,
            raw_pub_key=self.cryptosystem.pub_key,
            precision=self.encode_precision,
            pool=self.encode_pool,
        )
        enc_z_tilde = cipher_matmul(
            cipher_matrix=enc_a,
            plain_matrix=w_tilde_encode,
            executor_pool=self.thread_pool,
            scheduler_pool=self.scheduler_pool,
        )
        # enc_z_tilde = np.matmul(enc_a, w_tilde_encode) # shape: (bs, output_nodes)
        self.messenger.send(enc_z_tilde)

    def fed_backward(self, grad):
        grad = grad.cpu().numpy()
        grad_encode = encode(
            raw_data=grad,
            raw_pub_key=self.cryptosystem.pub_key,
            precision=self.encode_precision,
            pool=self.encode_pool,
        )
        enc_w_tilde_grad = cipher_matmul(
            cipher_matrix=self.curr_enc_a.transpose(),
            plain_matrix=grad_encode,
            executor_pool=self.thread_pool,
            scheduler_pool=self.scheduler_pool,
        )
        # enc_w_tilde_grad = np.matmul(self.curr_enc_a.transpose(), grad_encode)
        self.messenger.send(enc_w_tilde_grad)

        w_tilde_grad, enc_w_acc = self.messenger.recv()

        enc_a_grad_noise = cipher_matmul(
            cipher_matrix=enc_w_acc,
            plain_matrix=grad_encode.transpose(),
            executor_pool=self.thread_pool,
            scheduler_pool=self.scheduler_pool,
        )
        enc_a_grad_noise = enc_a_grad_noise.transpose()  # shape: (batch_size, in_nodes)
        # TODO: optimize it
        enc_a_grad = np.matmul(grad, self.w_tilde.transpose()) - enc_a_grad_noise

        # enc_a_grad = np.matmul(grad, self.w_tilde.transpose()) \
        #              - np.matmul(grad_encode, enc_w_acc.transpose())
        self.w_tilde = self.w_tilde - self.eta * w_tilde_grad

        return enc_a_grad

    def close_pool(self):
        if self.thread_pool is not None:
            self.thread_pool.close()
            self.scheduler_pool.close()
            self.encode_pool.close()
