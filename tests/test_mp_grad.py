import os
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import gmpy2
import numpy as np
from line_profiler_pycharm import profile
from phe import EncodedNumber, EncryptedNumber, generate_paillier_keypair
from tqdm import tqdm, trange

from linkefl.crypto import fast_add_ciphers, fast_mul_ciphers


def encode(trainset, public_key, prec=0.001):
    trainset_encode = []
    num_samples = trainset.shape[0]
    for k in range(num_samples):
        row = [
            EncodedNumber.encode(public_key, val, precision=prec) for val in trainset[k]
        ]
        trainset_encode.append(row)
    return np.array(trainset_encode)


def _target_grad_pool(*args):
    return sum(args) * (-1.0 / len(args))


def _target_grad_add(enc_grads, avg_grads, bs, start, end, thread_pool):
    for k in range(start, end):
        grad = fast_add_ciphers(enc_grads[k], thread_pool)
        avg_grads[k] = grad * (-1.0 / bs)
    return True


def _target_grad_mul(xencode, encres, encgrads, start, end, thread_pool):
    for k in range(start, end):
        grad = fast_mul_ciphers(xencode[k], encres[k], thread_pool)
        encgrads[k] = grad
    return True


def grad_mp_pool(
    x_encode,
    pub_key,
    params,
    enc_residue,
    batch_idxes,
    mul_serial=False,
    add_with_mp=True,
    executor_pool=None,
    scheduler_pool=None,
):
    n_batch_samples, n_batch_features = len(batch_idxes), params.size

    # 1. collect encrypted gradient items
    if mul_serial:
        n = pub_key.n
        n_squared = pub_key.n**2
        max_int = pub_key.max_int
        r_ciphers = [enc_r.ciphertext(False) for enc_r in enc_residue]
        r_ciphers_neg = [gmpy2.invert(r_cipher, n_squared) for r_cipher in r_ciphers]
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
                enc_train_grads[j].append(enc_j)  # a python list
    else:
        enc_train_grads = [None] * n_batch_samples
        data_size = n_batch_samples
        n_schedulers = scheduler_pool._processes
        quotient = data_size // n_schedulers
        remainder = data_size % n_schedulers
        async_results = []
        for idx in range(n_schedulers):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_schedulers - 1:
                end += remainder
            result = scheduler_pool.apply_async(
                _target_grad_mul,
                args=(
                    x_encode,
                    enc_residue,
                    enc_train_grads,
                    start,
                    end,
                    executor_pool,
                ),
            )
            async_results.append(result)
        for result in async_results:
            assert result.get() is True
        enc_train_grads = np.array(enc_train_grads).transpose()

    # 2. add encrypted gradients
    if add_with_mp:
        avg_grads = executor_pool.starmap(_target_grad_pool, enc_train_grads)
    else:
        bs = n_batch_samples
        n_schedulers = scheduler_pool._processes
        data_size = n_batch_features
        avg_grads = [None] * data_size
        quotient = data_size // n_schedulers
        remainder = data_size % n_schedulers
        async_results = []
        for idx in range(n_schedulers):
            start = idx * quotient
            end = (idx + 1) * quotient
            if idx == n_schedulers - 1:
                end += remainder
            result = scheduler_pool.apply_async(
                _target_grad_add,
                args=(enc_train_grads, avg_grads, bs, start, end, executor_pool),
            )
            async_results.append(result)
        for result in async_results:
            assert result.get() is True

    # 3. return the results
    return np.array(avg_grads)


def plaintext_grad(trainset, num_batches, bs):
    for k in range(num_batches):
        batch_idxes = np.arange(k * bs, (k + 1) * bs)
        res = np.random.rand(bs)
        grad = -1 * (res[:, np.newaxis] * trainset[batch_idxes]).mean(  # noqa: F841
            axis=0
        )


def enc_grad_serial(trainset, num_batches, bs, encrypted_zero):
    for k in trange(num_batches):
        batch_idxes = np.arange(k * bs, (k + 1) * bs)
        np.random.seed(k)
        res = np.random.rand(bs)
        encrypted_res = np.array([val + encrypted_zero for val in res])
        encrypted_grad = -1 * (  # noqa: F841
            encrypted_res[:, np.newaxis] * trainset[batch_idxes]
        ).mean(axis=0)
        # print([priv_key.decrypt(val) for val in enc_grad])
        # break


if __name__ == "__main__":
    pub_key_, priv_key = generate_paillier_keypair(n_length=1024)
    n_samples, n_features = 30000, 100
    batch_size = 100
    precision = 0.001
    n_batches = n_samples // batch_size
    enc_zero = pub_key_.encrypt(0)

    x_train = np.random.rand(n_samples, n_features)
    start_time = time.time()
    print("encoding x_train...")
    x_encode_ = encode(x_train, public_key=pub_key_, prec=precision)
    print("done.")
    print("encoding time: {}".format(time.time() - start_time))
    model_params = np.random.rand(n_features)

    # 1. compute plaintext gradient
    start_time = time.time()
    plaintext_grad(trainset=x_train, num_batches=n_batches, bs=batch_size)
    print("plaintext gradient elapsed time: {}".format(time.time() - start_time))

    # 2. compute encrypted gradient
    start_time = time.time()
    enc_grad_serial(
        trainset=x_train, num_batches=n_batches, bs=batch_size, encrypted_zero=enc_zero
    )
    print("encrypted gradient elapsed time: {}".format(time.time() - start_time))

    # 3. our methods
    schedule_pool = ThreadPool(4)
    process_pool = Pool(os.cpu_count())
    thread_pool_ = ThreadPool(os.cpu_count())
    case = [True, True, process_pool]
    # case = [True, False, thread_pool]
    # case = [False, False, thread_pool]

    start_time = time.time()
    for i in trange(n_batches):
        curr_batch_idxs = np.arange(i * batch_size, (i + 1) * batch_size)
        np.random.seed(i)
        residue = np.random.rand(batch_size)
        enc_residue_ = np.array([val + enc_zero for val in residue])
        enc_grad = grad_mp_pool(
            x_encode=x_encode_,
            pub_key=pub_key_,
            params=model_params,
            enc_residue=enc_residue_,
            batch_idxes=curr_batch_idxs,
            mul_serial=case[0],
            add_with_mp=case[1],
            executor_pool=case[2],
            scheduler_pool=schedule_pool,
        )
        # print([priv_key.decrypt(val) for val in enc_grad])
        # break
    print("our method elapsed time: {}".format(time.time() - start_time))
    schedule_pool.close()
    process_pool.close()
    thread_pool_.close()
