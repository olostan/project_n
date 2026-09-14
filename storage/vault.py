"""
Project N: Local AES-256-GCM Two-Key Vault & Envelope Encryption.
Implements per-clip random DEKs wrapped under a Key Encryption Key (KEK)
with unattended macOS Data Protection Keychain semantics and mock provider for testing.
"""

import os
from pathlib import Path
from typing import Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class KeychainProvider(Protocol):
    """Abstract interface for hardware/system keychain key management."""

    def get_ingest_kek(self) -> bytes:
        """Returns the 256-bit Key Encryption Key for background unattended processing."""
        ...

    def get_review_kek(self) -> bytes:
        """Returns the 256-bit Key Encryption Key for interactive review / export."""
        ...


class MockKeychainProvider:
    """In-memory deterministic keychain provider for hermetic automated testing."""

    def __init__(self, seed_key: bytes | None = None) -> None:
        if seed_key is None:
            self._ingest_kek = AESGCM.generate_key(bit_length=256)
            self._review_kek = AESGCM.generate_key(bit_length=256)
        else:
            self._ingest_kek = seed_key[:32].ljust(32, b"\x01")
            self._review_kek = seed_key[:32].ljust(32, b"\x02")

    def get_ingest_kek(self) -> bytes:
        return self._ingest_kek

    def get_review_kek(self) -> bytes:
        return self._review_kek


class VaultManager:
    """
    Manages local AES-256-GCM envelope encryption, DEK generation,
    ciphertext decryption, and cryptographic erasure.
    """

    def __init__(
        self,
        keychain: KeychainProvider | None = None,
        vault_dir: Path | None = None,
    ) -> None:
        self.keychain = keychain if keychain is not None else MockKeychainProvider()
        self.vault_dir = vault_dir
        self._memory_store: dict[str, tuple[bytes, bytes, bytes]] = {}

    def store_clip(self, clip_id: str, raw_data: bytes) -> None:
        """Encrypts and persists clip media under envelope encryption."""
        ciphertext, wrapped_dek, nonce = self.encrypt_clip(raw_data, clip_id=clip_id)
        if self.vault_dir:
            self.vault_dir.mkdir(parents=True, exist_ok=True)
            clip_file = self.vault_dir / f"{clip_id}.enc"
            with open(clip_file, "wb") as f:
                f.write(len(wrapped_dek).to_bytes(4, "big") + wrapped_dek + nonce + ciphertext)
        else:
            self._memory_store[clip_id] = (ciphertext, wrapped_dek, nonce)

    def retrieve_clip(self, clip_id: str) -> bytes:
        """Retrieves and decrypts stored clip media."""
        if self.vault_dir:
            clip_file = self.vault_dir / f"{clip_id}.enc"
            if not clip_file.exists():
                raise FileNotFoundError(f"Encrypted clip {clip_id} not found in vault.")
            data = clip_file.read_bytes()
            dek_len = int.from_bytes(data[:4], "big")
            wrapped_dek = data[4 : 4 + dek_len]
            nonce = data[4 + dek_len : 4 + dek_len + 12]
            ciphertext = data[4 + dek_len + 12 :]
            return self.decrypt_clip(ciphertext, wrapped_dek, nonce, clip_id=clip_id)
        elif clip_id in self._memory_store:
            ciphertext, wrapped_dek, nonce = self._memory_store[clip_id]
            return self.decrypt_clip(ciphertext, wrapped_dek, nonce, clip_id=clip_id)
        else:
            raise FileNotFoundError(f"Encrypted clip {clip_id} not found in vault.")

    def has_clip(self, clip_id: str) -> bool:
        """Checks if encrypted clip exists in vault."""
        if self.vault_dir:
            return (self.vault_dir / f"{clip_id}.enc").exists()
        return clip_id in self._memory_store

    def erase_stored_clip(self, clip_id: str) -> None:
        """Cryptographically erases a stored clip."""
        if self.vault_dir:
            clip_file = self.vault_dir / f"{clip_id}.enc"
            if clip_file.exists():
                self.erase_clip(clip_file)
        self._memory_store.pop(clip_id, None)

    def encrypt_clip(
        self,
        raw_data: bytes,
        clip_id: str,
    ) -> tuple[bytes, bytes, bytes]:
        """
        Encrypts clip data using a fresh per-clip random DEK wrapped under ingest KEK.

        Args:
            raw_data: Raw video/audio payload.
            clip_id: Unique clip identifier used as Authenticated Additional Data (AAD).

        Returns:
            tuple (ciphertext, wrapped_dek, nonce):
                - ciphertext: AES-256-GCM encrypted media bytes.
                - wrapped_dek: Per-clip DEK encrypted under Ingest KEK.
                - nonce: 12-byte initialization vector.
        """
        # 1. Generate random 256-bit DEK
        dek = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)

        # 2. Encrypt raw media with DEK and AAD
        cipher_dek = AESGCM(dek)
        ciphertext = cipher_dek.encrypt(nonce, raw_data, clip_id.encode("utf-8"))

        # 3. Wrap DEK under Ingest KEK
        ingest_kek = self.keychain.get_ingest_kek()
        kek_cipher = AESGCM(ingest_kek)
        kek_nonce = os.urandom(12)
        wrapped_dek = kek_nonce + kek_cipher.encrypt(kek_nonce, dek, clip_id.encode("utf-8"))

        return ciphertext, wrapped_dek, nonce

    def decrypt_clip(
        self,
        ciphertext: bytes,
        wrapped_dek: bytes,
        nonce: bytes,
        clip_id: str,
    ) -> bytes:
        """
        Unwraps DEK and decrypts clip ciphertext.

        Args:
            ciphertext: AES-256-GCM encrypted media bytes.
            wrapped_dek: Encrypted DEK (12-byte nonce + ciphertext).
            nonce: 12-byte media ciphertext nonce.
            clip_id: Unique clip identifier for AAD verification.

        Returns:
            Decrypted raw media bytes.
        """
        # 1. Unwrap DEK using Ingest KEK
        ingest_kek = self.keychain.get_ingest_kek()
        kek_cipher = AESGCM(ingest_kek)
        kek_nonce = wrapped_dek[:12]
        dek_ciphertext = wrapped_dek[12:]
        dek = kek_cipher.decrypt(kek_nonce, dek_ciphertext, clip_id.encode("utf-8"))

        # 2. Decrypt media with unwrapped DEK
        cipher_dek = AESGCM(dek)
        raw_data = cipher_dek.decrypt(nonce, ciphertext, clip_id.encode("utf-8"))
        return raw_data

    @staticmethod
    def erase_clip(file_path: Path) -> None:
        """
        Cryptographic and physical file erasure: overwrites with zeros and unlinks.
        """
        if not file_path.exists():
            return
        size = file_path.stat().st_size
        with open(file_path, "wb") as f:
            f.write(b"\x00" * size)
            f.flush()
            os.fsync(f.fileno())
        file_path.unlink()
