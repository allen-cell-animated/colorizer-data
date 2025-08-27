#!/usr/bin/env python3
"""
Drop-in replacement for `python -m http.server`, but over HTTPS.

Usage:
    python -m https_server [port] [--directory DIR]

Default port is 4443.
"""

import sys
import ssl
import http.server
import tempfile
import atexit
import os
import datetime
import argparse
from functools import partial

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def generate_self_signed_cert():
    """Generate a temporary self-signed cert + key and return their file paths."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )

    cert_file = tempfile.NamedTemporaryFile(delete=False)
    key_file = tempfile.NamedTemporaryFile(delete=False)

    atexit.register(os.remove, cert_file.name)
    atexit.register(os.remove, key_file.name)

    cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
    cert_file.close()

    key_file.write(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    key_file.close()

    return cert_file.name, key_file.name


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        http.server.SimpleHTTPRequestHandler.end_headers(self)
        if self.request_version != "HTTP/0.9":
            self._headers_buffer.append(b"\r\n")
            self.flush_headers()


def main():
    parser = argparse.ArgumentParser(
        description="Simple HTTPS server (self-signed cert)."
    )
    parser.add_argument(
        "port",
        type=int,
        nargs="?",
        default=4443,
        help="Port to serve on (default: 4443)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        default=".",
        help="Directory to serve (default: current dir)",
    )
    args = parser.parse_args()

    cert_path, key_path = generate_self_signed_cert()

    handler_class = partial(CORSRequestHandler, directory=args.directory)
    httpd = http.server.HTTPServer(("0.0.0.0", args.port), handler_class)

    # Modern SSL API
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"Serving HTTPS on https://0.0.0.0:{args.port} (directory: {args.directory})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
