# pf257_egkem_spnctr_v1_x25519.py
"""
PF257-EGKEM-SPNCTR v1 (X25519 KEM edition)

Hybrid image encryption with:
  - Pixel layer: SPN + forward/backward cumulative diffusion over Z_256
  - Block mixing: Hill-like affine mixing over prime field Z_257
  - Stream masking: CTR-like field stream over Z_257 (addition modulo 257)
  - Key transport: X25519 (ECDH) KEM-DEM (ephemeral-static ECDH)
  - Integrity: HMAC-SHA256 over AD + ciphertext blocks + wrapped parameters
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, TypedDict, Union
import hashlib
import hmac
import secrets
import struct

import numpy as np

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


BytesLike = Union[bytes, bytearray, memoryview]


class CipherPacket(TypedDict):
    # KEM header (X25519)
    kem_epk: bytes        # Alice ephemeral public key (32 bytes, raw)
    salt: bytes           # 32 bytes HKDF salt
    nonce: int            # 64-bit nonce for stream masking
    bs: int               # block size

    # Wrapped parameters (DEM ciphertext)
    enc_params: bytes     # encrypted (seed16,Kf,Kr,key32,gk[bs])

    # Ciphertext blocks (field values in 0..256 stored as uint16)
    blocks: np.ndarray    # shape: (ni, nj, bs, bs), dtype uint16, values 0..256
    orig_shape: Tuple[int, int]  # (H, W) original image shape
    pad_shape: Tuple[int, int]   # (Hp, Wp) padded shape

    # Authentication
    tag: Optional[bytes]  # HMAC-SHA256


class PF257_EGKEM_SPNCTR:
    P_FIELD: int = 257
    MOD256: int = 256

    def __init__(self, bs: int = 3, *, auth: bool = True) -> None:
        self.bs = int(bs)
        if self.bs < 1:
            raise ValueError("Block size (bs) must be at least 1.")
        # NOTE: This is Encrypt-then-MAC (EtM) authentication (HMAC-SHA256),
        # not a standardized AEAD mode like AES-GCM / ChaCha20-Poly1305.
        self._auth_enabled = bool(auth)
    # ---------------- HKDF (SHA-256) ----------------
    @staticmethod
    def _hkdf(ikm: bytes, info: bytes, length: int, *, salt: bytes) -> bytes:
        hlen = 32
        prk = hmac.new(salt or b"\0" * hlen, ikm, hashlib.sha256).digest()
        okm = bytearray()
        t = b""
        c = 1
        while len(okm) < length:
            t = hmac.new(prk, t + info + bytes([c]), hashlib.sha256).digest()
            okm.extend(t)
            c += 1
        return bytes(okm[:length])

    # ---------------- X25519 KEM ----------------
    @staticmethod
    def kem_generate_keypair() -> Tuple[x25519.X25519PrivateKey, bytes]:
        priv = x25519.X25519PrivateKey.generate()
        pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return priv, pub

    @staticmethod
    def _kem_encapsulate(bob_pub_raw: bytes) -> Tuple[bytes, bytes]:
        if not isinstance(bob_pub_raw, (bytes, bytearray)) or len(bob_pub_raw) != 32:
            raise ValueError("bob_pub_raw must be 32-byte X25519 public key (raw).")
        bob_pub = x25519.X25519PublicKey.from_public_bytes(bytes(bob_pub_raw))

        eph_priv = x25519.X25519PrivateKey.generate()
        eph_pub_raw = eph_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        shared = eph_priv.exchange(bob_pub)  # 32 bytes
        return eph_pub_raw, shared

    @staticmethod
    def _kem_decapsulate(bob_priv: x25519.X25519PrivateKey, kem_epk_raw: bytes) -> bytes:
        if not isinstance(kem_epk_raw, (bytes, bytearray)) or len(kem_epk_raw) != 32:
            raise ValueError("kem_epk_raw must be 32-byte X25519 public key (raw).")
        if not isinstance(bob_priv, x25519.X25519PrivateKey):
            raise TypeError("bob_priv must be X25519PrivateKey.")
        eph_pub = x25519.X25519PublicKey.from_public_bytes(bytes(kem_epk_raw))
        return bob_priv.exchange(eph_pub)

    # ---- context binding ----
    @staticmethod
    def _ctx(bob_pub_raw: bytes, kem_epk_raw: bytes) -> bytes:
        return b"CTX|" + hashlib.sha256(bytes(bob_pub_raw) + bytes(kem_epk_raw)).digest()

    # ---------------- Param packing ----------------
    def _pack_params(self, seed16: bytes, Kf: int, Kr: int, key32: bytes, gk: List[int]) -> bytes:
        if len(seed16) != 16:
            raise ValueError("seed16 must be 16 bytes.")
        if len(key32) != 32:
            raise ValueError("key32 must be 32 bytes.")
        if len(gk) != self.bs:
            raise ValueError("gk length mismatch.")
        gk_bytes = bytes([int(x) & 0xFF for x in gk])
        return seed16 + bytes([Kf & 0xFF, Kr & 0xFF]) + key32 + gk_bytes

    def _unpack_params(self, plain: bytes):
        need = 50 + self.bs
        if len(plain) != need:
            raise ValueError("Invalid params length.")
        seed16 = plain[:16]
        Kf = int(plain[16])
        Kr = int(plain[17])
        key32 = plain[18:50]
        g_raw = np.frombuffer(plain[50:], dtype=np.uint8)
        if np.any(g_raw == 0):
            raise ValueError("Invalid gk: zero coefficient not allowed.")
        gk_vec = g_raw.astype(np.uint16)

        seed = int.from_bytes(seed16, "big")
        rng = np.random.default_rng(seed)
        S = rng.permutation(256).astype(np.uint8)
        INV = np.empty_like(S)
        INV[S] = np.arange(256, dtype=np.uint8)
        return S, INV, Kf, Kr, key32, gk_vec

    def _wrap_params(self, shared: bytes, salt: bytes, ctx: bytes, plain: bytes) -> bytes:
        wrap_key = self._hkdf(shared, b"WRAP-PARAMS|" + ctx, 32, salt=salt)
        ks = self._bytes_stream(wrap_key, len(plain), nonce=0)
        return bytes(a ^ b for a, b in zip(plain, ks))

    def _unwrap_params(self, shared: bytes, salt: bytes, ctx: bytes, enc: bytes) -> bytes:
        wrap_key = self._hkdf(shared, b"WRAP-PARAMS|" + ctx, 32, salt=salt)
        ks = self._bytes_stream(wrap_key, len(enc), nonce=0)
        return bytes(a ^ b for a, b in zip(enc, ks))

    # ---------------- Streams ----------------
    @staticmethod
    def _bytes_stream(key32: bytes, n: int, nonce: int) -> bytes:
        if len(key32) != 32:
            raise ValueError("key32 must be 32 bytes.")
        blocks = (n + 31) // 32
        nb = int(nonce).to_bytes(8, "big", signed=False)
        out = bytearray(blocks * 32)
        pos = 0
        for ctr in range(blocks):
            msg = nb + struct.pack(">Q", ctr)
            digest = hmac.new(key32, msg, hashlib.sha256).digest()
            out[pos:pos + 32] = digest
            pos += 32
        return bytes(out[:n])

    def _field_stream(self, key32: bytes, n: int, nonce: int) -> np.ndarray:
        if len(key32) != 32:
            raise ValueError("key32 must be 32 bytes.")
        nb = int(nonce).to_bytes(8, "big", signed=False)
        out = np.empty(int(n), dtype=np.uint16)
        filled = 0
        ctr = 0
        while filled < int(n):
            digest = hmac.new(key32, nb + struct.pack(">Q", ctr), hashlib.sha256).digest()
            ctr += 1
            u16 = np.frombuffer(digest, dtype=">u2")
            u16 = u16[u16 != 65535]
            if u16.size == 0:
                continue
            vals = (u16 % self.P_FIELD).astype(np.uint16, copy=False)
            take = min(int(vals.size), int(n) - filled)
            out[filled:filled + take] = vals[:take]
            filled += take
        return out

    # ---------------- Pixel Layer (Z_256) ----------------
    def _encrypt_pixels(self, img_u8: np.ndarray, S: np.ndarray, Kf: int, Kr: int) -> np.ndarray:
        H, W = img_u8.shape
        N = H * W
        x = np.ascontiguousarray(img_u8, dtype=np.uint8).ravel()

        inv_lut = np.arange(255, -1, -1, dtype=np.uint8)
        x_sub = S[inv_lut[x]]

        if N == 0:
            return np.empty((H, W), dtype=np.uint8)

        c1 = (Kf + np.cumsum(x_sub, dtype=np.uint64)) % 256
        c1_u8 = c1.astype(np.uint8)
        c1_scr = S[c1_u8]

        c1_rev = c1_scr[::-1].astype(np.uint64)
        c2_rev = (Kr + np.cumsum(c1_rev, dtype=np.uint64)) % 256
        return c2_rev[::-1].astype(np.uint8).reshape(H, W)

    def _decrypt_pixels(self, c2_u8: np.ndarray, INV: np.ndarray, Kf: int, Kr: int) -> np.ndarray:
        H, W = c2_u8.shape
        N = H * W
        if N == 0:
            return np.empty((H, W), dtype=np.uint8)

        c2 = c2_u8.ravel().astype(np.uint8)
        c2_int = c2.astype(np.int32)[::-1]

        c1_scr_rev = (c2_int - np.roll(c2_int, 1)) % 256
        c1_scr_rev[0] = (c2_int[0] - Kr) % 256
        c1_scr = c1_scr_rev[::-1].astype(np.uint8)

        c1_u8 = INV[c1_scr].astype(np.int32)

        x_rec = (c1_u8 - np.roll(c1_u8, 1)) % 256
        x_rec[0] = (c1_u8[0] - Kf) % 256
        x_u8 = x_rec.astype(np.uint8)

        inv_lut = np.arange(255, -1, -1, dtype=np.uint8)
        return inv_lut[INV[x_u8]].reshape(H, W)

    # ---------------- Block Layer (Z_257) ----------------
    def _pad_edge_u8(self, img: np.ndarray, Hp: int, Wp: int) -> np.ndarray:
        H, W = img.shape
        pad_h = int(Hp) - int(H)
        pad_w = int(Wp) - int(W)
        if pad_h == 0 and pad_w == 0:
            return img
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")

    def _blocks_of(self, Q: np.ndarray) -> np.ndarray:
        Hp, Wp = Q.shape
        bs = self.bs
        if Hp % bs != 0 or Wp % bs != 0:
            raise ValueError("Padded shape must be multiples of bs.")
        ni = Hp // bs
        nj = Wp // bs
        return Q.reshape(ni, bs, nj, bs).swapaxes(1, 2).astype(np.uint16, copy=False)

    def _unblocks(self, blk: np.ndarray, Hp: int, Wp: int) -> np.ndarray:
        Q = blk.swapaxes(1, 2).reshape(int(Hp), int(Wp))
        return Q.astype(np.uint16, copy=False)

    @staticmethod
    def _vu_from_gk(gk_vec_u16: np.ndarray):
        V = np.diag(gk_vec_u16.astype(np.uint16))
        U = np.roll(V, 1, axis=1)
        return V, U

    def _mix_affine(self, Pblk_flat: np.ndarray, V: np.ndarray, U: np.ndarray) -> np.ndarray:
        p = self.P_FIELD
        diag = np.diag(V).astype(np.int32)
        P = Pblk_flat.astype(np.int32)
        C = (P * diag[None, None, :]) % p
        C = (C + U[None, :, :].astype(np.int32)) % p
        return C.astype(np.uint16, copy=False)

    def _unmix_affine(self, Cblk_flat: np.ndarray, V: np.ndarray, U: np.ndarray) -> np.ndarray:
        p = self.P_FIELD
        diag = np.diag(V).astype(np.int32)
        inv_diag = np.array([pow(int(v), p - 2, p) for v in diag], dtype=np.int32)
        D = (Cblk_flat.astype(np.int32) - U[None, :, :].astype(np.int32)) % p
        P = (D * inv_diag[None, None, :]) % p
        return P.astype(np.uint16, copy=False)

    # ---------------- Integrity (HMAC) ----------------
    @staticmethod
    def _lb(x: bytes) -> bytes:
        return len(x).to_bytes(4, "big") + x

    def _serialize_ad(self, packet: CipherPacket) -> bytes:
        ver = b"PF257-EGKEM-SPNCTR|v1|X25519|"
        nonce_b = int(packet["nonce"]).to_bytes(8, "big", signed=False)
        bs_b = int(packet["bs"]).to_bytes(4, "big", signed=False)
        H, W = packet["orig_shape"]
        Hp, Wp = packet["pad_shape"]
        shapes = (
            int(H).to_bytes(4, "big") + int(W).to_bytes(4, "big") +
            int(Hp).to_bytes(4, "big") + int(Wp).to_bytes(4, "big")
        )
        return (
            ver +
            self._lb(packet["kem_epk"]) +
            self._lb(packet["salt"]) +
            self._lb(nonce_b) +
            self._lb(bs_b) +
            self._lb(shapes) +
            self._lb(packet["enc_params"])
        )

    def _mac_key(self, shared: bytes, salt: bytes, ctx: bytes) -> bytes:
        return self._hkdf(shared, b"MACKEY|" + ctx, 32, salt=salt)

    def _tag(self, shared: bytes, packet: CipherPacket, ctx: bytes) -> bytes:
        key = self._mac_key(shared, packet["salt"], ctx)
        h = hmac.new(key, self._serialize_ad(packet), hashlib.sha256)
        blocks_le = packet["blocks"].astype("<u2", copy=False)
        h.update(blocks_le.tobytes(order="C"))
        return h.digest()

    def _verify(self, shared: bytes, packet: CipherPacket, ctx: bytes) -> bool:
        tag = packet.get("tag")
        if not isinstance(tag, (bytes, bytearray)):
            return False
        calc = self._tag(shared, packet, ctx)
        return hmac.compare_digest(tag, calc)

    # ================= PUBLIC API =================
    def encrypt(self, img: np.ndarray, *, bob_pub_raw: bytes, nonce: int,
                debug_return_field_image: bool = False):
        img_u8 = np.asarray(img, dtype=np.uint8)
        if img_u8.ndim != 2:
            raise ValueError("Input must be a 2D grayscale uint8 image.")
        H, W = img_u8.shape

        kem_epk, shared = self._kem_encapsulate(bob_pub_raw)
        salt = secrets.token_bytes(32)
        ctx = self._ctx(bob_pub_raw=bob_pub_raw, kem_epk_raw=kem_epk)

        seed16 = secrets.token_bytes(16)
        Kf = secrets.randbelow(256)
        Kr = secrets.randbelow(256)
        key32 = secrets.token_bytes(32)
        gk = [secrets.randbelow(255) + 1 for _ in range(self.bs)]
        plain_params = self._pack_params(seed16, Kf, Kr, key32, gk)
        enc_params = self._wrap_params(shared, salt, ctx, plain_params)

        S, INV, Kf2, Kr2, key32_2, gk_vec = self._unpack_params(plain_params)
        assert Kf2 == Kf and Kr2 == Kr and key32_2 == key32

        c2_u8 = self._encrypt_pixels(img_u8, S, Kf, Kr)

        bs = self.bs
        Hp = H + ((-H) % bs)
        Wp = W + ((-W) % bs)
        c2_pad_u8 = self._pad_edge_u8(c2_u8, Hp, Wp)
        c2_field = c2_pad_u8.astype(np.uint16, copy=False)

        blk = self._blocks_of(c2_field)
        ni, nj, _, _ = blk.shape
        blk_flat = blk.reshape((ni * nj, bs, bs))

        V, U = self._vu_from_gk(gk_vec)
        mixed_flat = self._mix_affine(blk_flat, V, U)
        mixed_blk = mixed_flat.reshape((ni, nj, bs, bs))
        mixed_field = self._unblocks(mixed_blk, Hp, Wp)

        stream = self._field_stream(key32, Hp * Wp, nonce).reshape(Hp, Wp).astype(np.int32)
        masked_field = ((mixed_field.astype(np.int32) + stream) % self.P_FIELD).astype(np.uint16)

        cblk = self._blocks_of(masked_field)

        packet: CipherPacket = {
            "kem_epk": kem_epk,
            "salt": salt,
            "nonce": int(nonce),
            "bs": int(bs),
            "enc_params": enc_params,
            "blocks": cblk,
            "orig_shape": (int(H), int(W)),
            "pad_shape": (int(Hp), int(Wp)),
            "tag": None,
        }
        if self._auth_enabled:
            packet["tag"] = self._tag(shared, packet, ctx)

        return packet, (masked_field if debug_return_field_image else cblk)

    def decrypt(self, packet: CipherPacket, *, bob_priv: x25519.X25519PrivateKey,
                verify: bool = True, bob_pub_raw: Optional[bytes] = None) -> np.ndarray:
        bs = int(packet["bs"])
        if bs != self.bs:
            raise ValueError(f"Block size mismatch: packet bs={bs}, instance bs={self.bs}")

        Hp, Wp = packet["pad_shape"]
        H, W = packet["orig_shape"]

        kem_epk = packet["kem_epk"]
        shared = self._kem_decapsulate(bob_priv, kem_epk)

        if bob_pub_raw is None:
            ctx = b"CTX|" + hashlib.sha256(bytes(kem_epk)).digest()
        else:
            if not isinstance(bob_pub_raw, (bytes, bytearray)) or len(bob_pub_raw) != 32:
                raise ValueError("bob_pub_raw must be 32-byte X25519 public key (raw).")
            ctx = self._ctx(bob_pub_raw=bob_pub_raw, kem_epk_raw=kem_epk)

        if self._auth_enabled and verify:
            if not self._verify(shared, packet, ctx):
                raise ValueError("Authentication failed: tampered packet or wrong key.")

        plain = self._unwrap_params(shared, packet["salt"], ctx, packet["enc_params"])
        S, INV, Kf, Kr, key32, gk_vec = self._unpack_params(plain)

        masked_field = self._unblocks(packet["blocks"], Hp, Wp).astype(np.uint16, copy=False)

        stream = self._field_stream(key32, Hp * Wp, packet["nonce"]).reshape(Hp, Wp).astype(np.int32)
        mixed_field = ((masked_field.astype(np.int32) - stream) % self.P_FIELD).astype(np.uint16)

        blk = self._blocks_of(mixed_field)
        ni, nj, _, _ = blk.shape
        blk_flat = blk.reshape((ni * nj, bs, bs))

        V, U = self._vu_from_gk(gk_vec)
        rec_flat = self._unmix_affine(blk_flat, V, U)
        rec_blk = rec_flat.reshape((ni, nj, bs, bs))
        c2_field_pad = self._unblocks(rec_blk, Hp, Wp)

        c2_u8 = (c2_field_pad[:H, :W] % 256).astype(np.uint8)
        return self._decrypt_pixels(c2_u8, INV, Kf, Kr)


if __name__ == "__main__":
    bob_priv, bob_pub = PF257_EGKEM_SPNCTR.kem_generate_keypair()
    cipher = PF257_EGKEM_SPNCTR(bs=5, auth=True)

    img = (np.arange(0, 256, dtype=np.uint8).reshape(16, 16))
    packet, _ = cipher.encrypt(img, bob_pub_raw=bob_pub, nonce=123456789)
    rec = cipher.decrypt(packet, bob_priv=bob_priv, verify=True, bob_pub_raw=bob_pub)

    assert np.array_equal(img, rec), "Decryption mismatch!"
    print("OK (X25519 KEM)")
