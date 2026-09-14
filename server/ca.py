"""
Project N: Local Certificate Authority (CA) for Companion Mutual Authentication.
Issues ECDSA P-256 client certificates for paired devices per docs/SPECS.md §5.1.
"""

import threading
from datetime import UTC, datetime, timedelta

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


class LocalCertificateAuthority:
    """In-memory or persistent local ECDSA Certificate Authority."""

    _instance: "LocalCertificateAuthority | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._private_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name(
            [
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Project N Local Daemon"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Project N Local Root CA"),
            ]
        )
        now = datetime.now(UTC)
        self._ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(self._private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=3650))
            .sign(self._private_key, hashes.SHA256())
        )

    @classmethod
    def get_instance(cls) -> "LocalCertificateAuthority":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def ca_cert_pem(self) -> str:
        return self._ca_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")

    def sign_csr(self, csr_pem: str, validity_days: int = 365) -> str:
        """
        Parses an ECDSA CSR and returns a signed client certificate PEM.
        Raises ValueError if CSR is malformed.
        """
        try:
            csr = x509.load_pem_x509_csr(csr_pem.encode("utf-8"))
        except Exception as err:
            raise ValueError(f"Invalid CSR PEM payload: {err}") from err

        now = datetime.now(UTC)
        client_cert = (
            x509.CertificateBuilder()
            .subject_name(csr.subject)
            .issuer_name(self._ca_cert.subject)
            .public_key(csr.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=validity_days))
            .sign(self._private_key, hashes.SHA256())
        )
        return client_cert.public_bytes(serialization.Encoding.PEM).decode("utf-8")


def generate_test_csr(
    device_id: str = "caregiver_device",
) -> tuple[str, ec.EllipticCurvePrivateKey]:
    """Utility to generate a valid ECDSA P-256 CSR for tests."""
    dev_key = ec.generate_private_key(ec.SECP256R1())
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(
            x509.Name(
                [
                    x509.NameAttribute(NameOID.COMMON_NAME, device_id),
                ]
            )
        )
        .sign(dev_key, hashes.SHA256())
    )
    csr_pem = csr.public_bytes(serialization.Encoding.PEM).decode("utf-8")
    return csr_pem, dev_key
