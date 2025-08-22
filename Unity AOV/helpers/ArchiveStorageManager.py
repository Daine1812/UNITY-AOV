# based on: https://github.com/Razmoth/PGRStudio/blob/master/AssetStudio/PGR/PGR.cs
from typing import Tuple, Union

from ..streams import EndianBinaryReader

from sm4 import SM4Key
 
class ArchiveStorageDecryptor:
    unknown_1: int
    index: bytes
    substitute: bytes = bytes(0x10)

    def __init__(self, reader: EndianBinaryReader) -> None:
        self.BlockSM4Key=b"\x02\x24\xDC\x74\x07\x1B\x94\x36\x25\x20\x0A\xD6\x14\x62\x05\xE3"
        self.BlockSM4IV=b'\x79\x7B\xCD\x5D\x7D\x7B\xB1\x11\x43\xD0\x0D\x71\x3C\xDA\xA8\x08'
        

    def decrypt_block(self, data: bytes):
        blocksize = (len(data)//16)*16
        unencrypted_part = data[-(len(data) % 16):] if len(data) % 16 != 0 else b''
        key0 = SM4Key(self.BlockSM4Key)
        decrypt_data=key0.decrypt(data[0:blocksize],initial=bytearray(self.BlockSM4IV))
        return decrypt_data + unencrypted_part
