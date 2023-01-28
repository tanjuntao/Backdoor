import json
import random
import secrets
import time

import gmpy2
from phe.paillier import (
    PaillierPrivateKey,
    PaillierPublicKey,
    generate_paillier_keypair,
)

from linkefl.crypto import RSA as LinkeRSA

BASE_URL = "http://10.7.7.150:10310/isafeidms"
AUTH_TOKEN_SDK = "sOslPLw84gJC0BoMjHUf9tcmEmWOdyX8tyHwt6xBnRo="
test_users = [["user" + str(i), "psw" + str(i)] for i in range(500)]
test_funcs = ["func" + str(i) for i in range(30)]
test_taskids = ["taskid" + str(i) for i in range(100)]


# =========================SDK================================
def idms_valid_cred(username, password, taskId, funcId):
    import requests

    url = "http://10.7.7.150:10310/isafeidms/auth/func/check"
    params = {
        "userName": username,
        "pwdHash": password,
        "taskId": taskId,
        "funcId": funcId,
    }
    headers = {"Content-Type": "application/json", "Token": AUTH_TOKEN_SDK}

    ret = requests.get(url, params=params, headers=headers)

    # print(ret.json())
    return ret.json()


# idms_valid_cred('user304', 'psw304', 'taskid61', 'func13')


def idms_get_keypair_rsa(username, password, taskId):
    import requests

    url = "http://10.7.7.150:10310/isafeidms/key/pair/get"
    params = {
        "userName": username,
        "pwdHash": password,
        "taskId": taskId,
        "type": "rsa",
    }
    headers = {"Content-Type": "application/json", "Token": AUTH_TOKEN_SDK}

    ret = requests.get(url, params=params, headers=headers)

    # print(ret.json())
    return ret.json()


# idms_get_keypair_rsa('user304', 'psw304', 'taskid61')


def idms_get_pubkey_rsa(username, password, taskId):
    import requests

    url = "http://10.7.7.150:10310/isafeidms/key/pub/get"
    params = {
        "userName": username,
        "pwdHash": password,
        "taskId": taskId,
        "type": "rsa",
    }
    headers = {"Content-Type": "application/json", "Token": AUTH_TOKEN_SDK}

    ret = requests.get(url, params=params, headers=headers)

    print(ret.json())
    return ret.json()


# idms_get_pubkey_rsa('user152', 'psw152', 'taskid61')


def idms_get_keypair_paillier(username, password, taskId):
    import requests

    url = "http://10.7.7.150:10310/isafeidms/key/pair/get"
    params = {
        "userName": username,
        "pwdHash": password,
        "taskId": taskId,
        "type": "paillier",
    }
    headers = {"Content-Type": "application/json", "Token": AUTH_TOKEN_SDK}

    ret = requests.get(url, params=params, headers=headers)

    print(ret.json())
    return ret.json()


# idms_get_keypair_paillier('user304', 'psw304', 'taskid61')


def idms_get_pubkey_paillier(username, password, taskId):
    import requests

    url = "http://10.7.7.150:10310/isafeidms/key/pub/get"
    params = {
        "userName": username,
        "pwdHash": password,
        "taskId": taskId,
        "type": "paillier",
    }
    headers = {"Content-Type": "application/json", "Token": AUTH_TOKEN_SDK}

    ret = requests.get(url, params=params, headers=headers)

    print(ret.json())
    return ret.json()


# idms_get_pubkey_paillier('user152', 'psw152', 'taskid61')

# =========================SDK================================


if __name__ == "__main__":
    """RSA validation"""
    key_size = 1024
    rsa_dict = idms_get_keypair_rsa("user304", "psw304", "taskid61")
    secret_key = json.loads(rsa_dict["data"]["secretKey"])
    e = int(secret_key["e"])
    n = int(secret_key["n"])
    d = int(secret_key["d"])

    rsa_crypto = LinkeRSA(key_size=1024, e=e)
    data = [random.randint(1, 10000) for _ in range(1000)]
    start_time = time.time()
    rsa_crypto.sign_vector(data, using_pool=True)
    print("multi-threading elapsed time: {}".format(time.time() - start_time))
    start_time = time.time()
    rsa_crypto.sign_vector(data, using_pool=True)
    print("single thread elapsed time: {}".format(time.time() - start_time))

    # priv_key = RSA.generate(key_size, e=e)
    # pub_key = RSA.construct((priv_key.n, priv_key.e))
    # start_time = time.time()
    # result = [gmpy2.powmod(random.randint(1, 10000000), d, n) for _ in range(1000)]
    # print('elapsed time: {}'.format(time.time() - start_time))

    """Paillier validation"""
    # paillier_dict = idms_get_keypair_paillier('user304', 'psw304', 'taskid61')
    # secret_key = json.loads(paillier_dict['data']['secretKey'])
    # n = int(secret_key['n'])
    # p = int(secret_key['p'])
    # q = int(secret_key['q'])

    # pub_key = PaillierPublicKey(n)
    # priv_key = PaillierPrivateKey(pub_key, p, q)
    # a = 100.1
    # enc_a = pub_key.encrypt(a)
    # dec_a = priv_key.decrypt(enc_a)
    # print(dec_a)
