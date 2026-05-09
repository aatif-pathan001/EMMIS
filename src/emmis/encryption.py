import base64
import hashlib
from emmis.config import settings

class Cipher:

    def __init__(self) -> None:
        
        self.xor_key = settings.XOR_KEY % 256
        self.shift_value = settings.SHIFT_VALUE % 256
        self.scramble_seed = settings.SCRAMBLE_SEED

    def _xor_transform(self, data: bytes) -> bytes:
        return bytes([b ^ (self.xor_key ^ (i % 256)) for i, b in enumerate(data)])

    def _shift_bytes(self, data: bytes, forward: bool = True) -> bytes:
        delta = self.shift_value if forward else -self.shift_value
        return bytes([(b + delta) % 256 for b in data])

    def _generate_permutation(self, length: int) -> list:
        indices = list(range(length))
        seed = self.scramble_seed
        for i in range(length - 1, 0, -1):
            seed = (seed * 1_664_525 + 1_013_904_223) & 0xFFFF_FFFF
            j = seed % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        return indices

    def _scramble(self, data: bytes) -> bytes:
        if not data:
            return data
        perm = self._generate_permutation(len(data))
        result = bytearray(len(data))
        for new_idx, old_idx in enumerate(perm):
            result[new_idx] = data[old_idx]
        return bytes(result)

    def _unscramble(self, data: bytes) -> bytes:
        if not data:
            return data
        perm = self._generate_permutation(len(data))
        result = bytearray(len(data))
        for new_idx, old_idx in enumerate(perm):
            result[old_idx] = data[new_idx]
        return bytes(result)

    def encrypt(self, data: bytes) -> str:
        data_xor = self._xor_transform(data)    
        data_shift = self._shift_bytes(data_xor, True) 
        data_scramble = self._scramble(data_shift)
        data_encrypt = base64.b64encode(data_scramble).decode("utf-8")         
        return data_encrypt

    def decrypt(self, encrypted: str) -> bytes:
        try:
            data_decrypt = base64.b64decode(encrypted.encode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid Base64 input: {exc}") from exc

        data_unscramble = self._unscramble(data_decrypt)          
        data_unshift = self._shift_bytes(data_unscramble, False)  
        data = self._xor_transform(data_unshift)
        return  data       
    
    def encrypt_text(self, text: str) -> str:
        return self.encrypt(text.encode("utf-8"))

    def decrypt_text(self, encrypted: str) -> str:
        return self.decrypt(encrypted).decode("utf-8")

    def encrypt_image(self, image_bytes: bytes) -> str:
        return self.encrypt(image_bytes)

    def decrypt_image(self, encrypted: str) -> bytes:
        return self.decrypt(encrypted)

    def checksum(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def verify_cipher(self, original: bytes) -> bool:
        return self.decrypt(self.encrypt(original)) == original
