import base64
import time
from base64 import b64decode, b64encode

import pyotp
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


def isafelinke_sparta_validlicense():
    def aes_encrypt(key, plaintext):
        cipher_ = AES.new(key, AES.MODE_ECB)
        return b64encode(cipher_.encrypt(pad(plaintext.encode(), 16))).decode()

    def aes_decrypt(key, ciphertext):
        cipher_ = AES.new(key, AES.MODE_ECB)
        return unpad(cipher_.decrypt(b64decode(ciphertext.encode())), 16).decode()

    _keySparta = "5oiR5piv5Y2O5ZOl5ZOIMQ=="
    keySparta = base64.b64decode(_keySparta)
    spartaOtpSeed = "4WA2LZN2W7SLTFXFV2O6JP454S6ZDZ4IXDTYROHFU2EOLJUI"
    danaID_py = "202212221116306920MPnf20Nz5Fw91c"
    otpSeedSDK_py = "467I5ZVUWLS2JJ7ITOD6LEUM4WA2LZN2W7SLTFXFV2OQ===="
    sparta_url_list = ["http://sparta.isafetech.com.cn:80/isafelinke/"]
    random_OtpParam = int(time.time())
    py_hotp = pyotp.HOTP(otpSeedSDK_py).at(random_OtpParam)
    raw_msgFromSDK = str(random_OtpParam) + "#" + str(danaID_py) + "#" + str(py_hotp)
    msgFromSDK = aes_encrypt(keySparta, raw_msgFromSDK)
    try:
        for oneurl in sparta_url_list:
            r = requests.post(oneurl, data=msgFromSDK, timeout=(10, 2))  # 链接5s，读取2s超时
            if "ERR" in str(r.content):
                return False
            try:
                cipher = r.content
                cipher = cipher.decode()
                infos = aes_decrypt(keySparta, cipher).split("#")
                sparta_htop_param = infos[0]
                if not type(eval(sparta_htop_param)) == int:
                    return False
                sparta_hotp = infos[1]
                real_sparta_otp = pyotp.HOTP(spartaOtpSeed).at(eval(sparta_htop_param))
                if str(real_sparta_otp) == str(sparta_hotp):
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False
