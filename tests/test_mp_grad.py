from heapq import nsmallest
import multiprocessing
import os
import time

import gmpy2
import numpy as np
from phe import EncryptedNumber, EncodedNumber, generate_paillier_keypair


def _target_grad(*args):
    return sum(args)


def encode(x_train, pub_key, precision=0.001):
    x_encode = []
    n_samples = x_train.shape[0]
    for i in range(n_samples):
        row = [EncodedNumber.encode(pub_key, val, precision=precision)
                for val in x_train[i]]
        x_encode.append(row)
    return np.array(x_encode)


def grad_mp_pool(x_encode, pub_key, params, enc_residue, batch_idxes, worker_pool):
    """compute encrypted gradients manully
    
    Args: 
        x_train: numpy array, the whole training dataset, the shape is (n_samples, n_features)
        pub_key, paillier public key
        params, numpy array, linear model parameter, the shape is (n_features, )
        enc_residue: python list, the length is batch_size
        batch_idxs: numpy array, the shape is (batch_size, )
        worker_pool, python multiprocessing Pool object
    
    Return:
        Encrypted gradient of current mini-batch 
    """
    n_batch_samples, n_batch_features = len(batch_idxes), params.size
    n = pub_key.n
    n_squared = pub_key.n ** 2
    max_int = pub_key.max_int
    r_ciphers = [enc_r.ciphertext(False) for enc_r in enc_residue]
    r_ciphers_neg = [gmpy2.invert(r_cipher, n_squared) for r_cipher in r_ciphers]

    # collect encrypted gradient items
    enc_train_grads = [[] for _ in range(n_batch_features)]
    for i in range(n_batch_samples):
        for j in range(n_batch_features):
            row = x_encode[batch_idxes[i]]
            encoding = row[j].encoding
            exponent = row[j].exponent + enc_residue[i].exponent
            if n - max_int < encoding:
                ciphertext = gmpy2.powmod(r_ciphers_neg[i], n - encoding, n_squared)
            else:
                ciphertext = gmpy2.powmod(r_ciphers[i], encoding, n_squared)
            enc_j = EncryptedNumber(pub_key, ciphertext, exponent)
            enc_train_grads[j].append(enc_j)

    # using pool to add encrypted gradients in parallel
    avg_grads = worker_pool.starmap(_target_grad, enc_train_grads)

    # average gradients
    for j in range(n_batch_features):
        avg_grads[j] = avg_grads[j] * (-1. / len(batch_idxes))

    # convert grads from python list to numpy.ndarray
    avg_grads = np.array(avg_grads)
    return avg_grads


if __name__ == '__main__':
    pub_key, priv_key = generate_paillier_keypair(n_length=1024)
    n_samples, n_features = 20000, 81
    batch_size = 100
    n_batches = n_samples // batch_size
    enc_zero = pub_key.encrypt(0)
    
    x_train = np.random.rand(n_samples, n_features)
    model_params = np.random.rand(n_features)

    # 1. compute plaintext gradient 
    start_time = time.time()
    for i in range(n_batches):
        curr_batch_idxs = np.arange(i * batch_size, (i+1) * batch_size)
        residue = np.random.rand(batch_size)
        plain_grad = -1 * (residue[:, np.newaxis] * x_train[curr_batch_idxs]).mean(axis=0)
    print('plaintext gradient elapsed time: {}'.format(time.time() - start_time))

    # 2. compute encrypted gradient
    start_time = time.time()
    for i in range(n_batches):
        # print(f"batch: {i}")
        curr_batch_idxs = np.arange(i * batch_size, (i+1) * batch_size)
        residue = np.random.rand(batch_size)
        # enc_residue = np.array([val + enc_zero for val in residue])
        enc_residue = np.array([pub_key.encrypt(val) for val in residue])
        enc_grad = -1 * (enc_residue[:, np.newaxis] * x_train[curr_batch_idxs]).mean(axis=0)
    print('encrypted gradient elapsed time: {}'.format(time.time() - start_time))


    # 3. our methods
    n_workers = os.cpu_count()
    pool = multiprocessing.Pool(n_workers)
    start_time = time.time()
    x_encode = encode(x_train, pub_key=pub_key)
    for i in range(n_batches):
        # print(f"batch : {i}")
        curr_batch_idxs = np.arange(i * batch_size, (i+1) * batch_size)
        residue = np.random.rand(batch_size)
        # enc_residue = np.array([val + enc_zero for val in residue])
        enc_residue = np.array([pub_key.encrypt(val) for val in residue])
        enc_grad = grad_mp_pool(
            x_encode=x_encode,
            pub_key=pub_key,
            params=model_params,
            enc_residue=enc_residue, 
            batch_idxes=curr_batch_idxs, 
            worker_pool=pool)
    print('our method elapsed time: {}'.format(time.time() - start_time))