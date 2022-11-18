import multiprocessing
import os

import numpy as np
from phe import EncodedNumber
import torch

from linkefl.common.const import Const
from linkefl.crypto import fast_add_ciphers, fast_mul_ciphers


class PassiveEncLayer:
    def __init__(self,
                 in_nodes,
                 out_nodes,
                 eta,
                 cryptosystem,
                 messenger,
                 num_workers=None,
                 random_state=None,
                 precision=0.001
    ):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.eta = eta
        self.cryptosystem = cryptosystem
        self.messenger = messenger
        self.precision = precision

        if num_workers is None:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.process_pool = multiprocessing.pool.Pool(self.num_workers)
        self.thread_pool = multiprocessing.pool.ThreadPool(self.num_workers)
        if cryptosystem.type == Const.PAILLIER:
            # using thread_pool for Paillier encrypt_vector
            self.enc_pool = self.thread_pool
        else:
            # using process_pool for FastPaillier encrypt_vector
            self.enc_pool = self.process_pool
        if random_state is not None:
            np.random.seed(random_state)
        self.w_acc = np.random.rand(in_nodes, out_nodes).astype(np.float32)

    def fed_forward(self, a):
        enc_a = self.cryptosystem.encrypt_vector(
            torch.flatten(a),
            True,
            self.num_workers,
            self.enc_pool
        ) # return a Python list
        enc_a = np.array(enc_a).reshape(a.shape) # convert it to numpy array
        self.messenger.send(enc_a)
        enc_z_tilde = self.messenger.recv() # numpy array, shape: (bs, output_nodes)
        z_tilde = self.cryptosystem.decrypt_vector(
            enc_z_tilde.flatten(),
            using_pool=True,
            thread_pool=self.thread_pool
        )
        z_tilde = [float(val) for val in z_tilde] # TODO: strange mpfr dtype
        z_tilde = torch.tensor(z_tilde).reshape(enc_z_tilde.shape)
        # z_tilde = np.array(z_tilde).reshape(enc_z_tilde.shape)
        z_clear = z_tilde - torch.matmul(a, torch.from_numpy(self.w_acc))
        return z_clear # tensor, shape: (bs, output_nodes)

    def fed_backward(self):
        enc_w_tilde_grad = self.messenger.recv() # numpy array, shape: (input_nodes, output_nodes)
        w_tilde_grad = self.cryptosystem.decrypt_vector(
            enc_w_tilde_grad.flatten(),
            using_pool=True,
            thread_pool=self.thread_pool
        )
        w_tilde_grad = [float(val) for val in w_tilde_grad] # TODO: strange mpfr dtype
        w_tilde_grad = np.array(w_tilde_grad).reshape(enc_w_tilde_grad.shape)

        w_curr = np.random.rand(self.in_nodes, self.out_nodes).astype(np.float32)
        w_tilde_grad = w_tilde_grad - w_curr / self.eta

        enc_w_acc = self.cryptosystem.encrypt_vector(
            self.w_acc.flatten(),
            True,
            self.num_workers,
            self.enc_pool
        )
        enc_w_acc = np.array(enc_w_acc).reshape(self.w_acc.shape)
        self.messenger.send([w_tilde_grad, enc_w_acc])

        # update w_acc
        self.w_acc = self.w_acc + w_curr

    def close_pool(self):
        self.process_pool.close()
        self.thread_pool.close()


class ActiveEncLayer:
    def __init__(self,
                 in_nodes,
                 out_nodes,
                 eta,
                 cryptosystem,
                 messenger,
                 num_workers=None,
                 random_state=None,
                 precision=0.001
    ):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.eta = eta
        self.cryptosystem = cryptosystem
        self.messenger = messenger
        self.precision = precision

        if num_workers is None:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.process_pool = multiprocessing.pool.Pool(self.num_workers)
        self.thread_pool = multiprocessing.pool.ThreadPool(self.num_workers)
        self.scheduler_pool = multiprocessing.pool.ThreadPool(self.num_workers)

        if random_state is not None:
            np.random.seed(random_state)
        self.w_tilde = np.random.rand(in_nodes, out_nodes).astype(np.float32)
        self.curr_enc_a = None

    def fed_forward(self):
        enc_a = self.messenger.recv() # is a numpy array, shape: (bs, input_nodes)
        self.curr_enc_a = enc_a
        w_tilde_encode = ActiveEncLayer._encode(
            self.w_tilde,
            self.cryptosystem.pub_key,
            self.precision
        )
        enc_z_tilde = ActiveEncLayer._cipher_matmul(
            cipher_matrix=enc_a,
            plain_matrix=w_tilde_encode,
            executor_pool=self.thread_pool,
            scheduler_pool=self.scheduler_pool
        )
        # enc_z_tilde = np.matmul(enc_a, w_tilde_encode) # shape: (bs, output_nodes)
        self.messenger.send(enc_z_tilde)

    def fed_backward(self, grad):
        grad = grad.numpy()
        grad_encode = ActiveEncLayer._encode(
            grad,
            self.cryptosystem.pub_key,
            self.precision
        )
        enc_w_tilde_grad = ActiveEncLayer._cipher_matmul(
            cipher_matrix=self.curr_enc_a.transpose(),
            plain_matrix=grad_encode,
            executor_pool=self.thread_pool,
            scheduler_pool=self.scheduler_pool
        )
        # enc_w_tilde_grad = np.matmul(self.curr_enc_a.transpose(), grad_encode)
        self.messenger.send(enc_w_tilde_grad)

        w_tilde_grad, enc_w_acc = self.messenger.recv()

        enc_a_grad_noise = ActiveEncLayer._cipher_matmul(
            cipher_matrix=enc_w_acc,
            plain_matrix=grad_encode.transpose(),
            executor_pool=self.thread_pool,
            scheduler_pool=self.scheduler_pool
        )
        enc_a_grad_noise = enc_a_grad_noise.transpose() # shape: (batch_size, in_nodes)
        # TODO: optimize it
        enc_a_grad = np.matmul(grad, self.w_tilde.transpose()) - enc_a_grad_noise

        # enc_a_grad = np.matmul(grad, self.w_tilde.transpose()) \
        #              - np.matmul(grad_encode, enc_w_acc.transpose())
        self.w_tilde = self.w_tilde - self.eta * w_tilde_grad

        return enc_a_grad

    @staticmethod
    def _cipher_matmul(cipher_matrix, plain_matrix, executor_pool, scheduler_pool):
        result_matrix = []
        for i in range(len(cipher_matrix)):
            curr_result = ActiveEncLayer._mat_vec_product(
                enc_vector=cipher_matrix[i],
                plain_matrix=plain_matrix,
                executor_pool=executor_pool,
                scheduler_pool=scheduler_pool
            )
            result_matrix.append(curr_result)

        return np.array(result_matrix)

    @staticmethod
    def _mat_vec_product(enc_vector, plain_matrix, executor_pool, scheduler_pool):
        height, width = plain_matrix.shape

        # 1. multiply each raw of plain_matrix with its corresponding enc value
        enc_result = [None] * height
        data_size = height
        n_schedulers = scheduler_pool._processes
        quotient = data_size // n_schedulers
        remainder = data_size % n_schedulers
        async_results = []
        for idx in range(n_schedulers):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_schedulers - 1:
                end += remainder
            # this will modify enc_result in place
            result = scheduler_pool.apply_async(
                ActiveEncLayer._target_grad_mul,
                args=(enc_vector, plain_matrix, enc_result, start, end, executor_pool)
            )
            async_results.append(result)
        for result in async_results:
            assert result.get() is True

        # 2. transpose enc_result
        enc_result = np.array(enc_result).transpose()

        # 3. average enc_result
        avg_result = [None] * width
        data_size = width
        n_schedulers = scheduler_pool._processes
        quotient = data_size // n_schedulers
        remainder = data_size % n_schedulers
        async_results = []
        for idx in range(n_schedulers):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_schedulers - 1:
                end += remainder
            # this will modify avg_result in place
            result = scheduler_pool.apply_async(
                ActiveEncLayer._target_grad_add,
                args=(enc_result, avg_result, start, end, executor_pool)
            )
            async_results.append(result)
        for result in async_results:
            assert result.get() is True

        return np.array(avg_result)

    @staticmethod
    def _target_grad_mul(enc_vector, plain_matrix, enc_result,
                         start, end, executor_pool):
        for k in range(start, end):
            enc_row = fast_mul_ciphers(
                plain_matrix[k],
                enc_vector[k],
                executor_pool
            )
            enc_result[k] = enc_row
        return True

    @staticmethod
    def _target_grad_add(enc_result, avg_result,
                         start, end, executor_pool):
        for k in range(start, end):
            row_sum = fast_add_ciphers(enc_result[k], executor_pool)
            avg_result[k] = row_sum
        return True

    @staticmethod
    def _encode(data: np.ndarray, pub_key, precision):
        data_flat = data.flatten()
        # remember to use val.item(), otherwise,
        # "TypeError('argument should be a string or a Rational instance'" will be raised
        data_encode = [EncodedNumber.encode(pub_key, val.item(), precision=precision)
                       for val in data_flat]
        return np.array(data_encode).reshape(data.shape)

    def close_pool(self):
        self.process_pool.close()
        self.thread_pool.close()
        self.scheduler_pool.close()
