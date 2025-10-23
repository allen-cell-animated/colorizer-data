#!/usr/bin/env python3

from http.server import HTTPServer, SimpleHTTPRequestHandler
from multiprocessing import Process
import argparse
import os
import signal
import socket
from time import sleep
from typing import List, Tuple
import webbrowser

# 6465 and 6470 are unassigned ports according to
# https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml
default_tfe_port = 6465
default_directory_port = 6470


def acquire_ports(default_ports: List[int]) -> List[Tuple[int, socket.socket]]:
    """
    Returns a list of tuples containing reserved port numbers and open sockets
    on those ports, attempting to use the provided list of defaults. If a
    default port is in use, returns another available port instead. The open
    sockets can be used to prevent other processes from acquiring the same
    ports.

    Note that the returned sockets must be closed by the caller when no longer
    needed.
    """
    ports = []
    for default_port in default_ports:
        if not (0 <= default_port <= 65535):
            raise ValueError("Port numbers must be between 0 and 65535.")
        try:
            # Try binding to the default port
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("localhost", default_port))
            ports.append((default_port, s))
        except OSError:
            # If the default port is in use, find an available port
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("localhost", 0))  # Bind to an available port
            ports.append((s.getsockname()[1], s))
    return ports


# Adapted from https://stackoverflow.com/a/21957017.
class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        SimpleHTTPRequestHandler.end_headers(self)


class ReactAppRequestHandler(CORSRequestHandler):
    def translate_path(self, path):
        # Serve the React app's index.html for any path that doesn't map to a
        # file for client-side routing.
        path = super().translate_path(path)
        if os.path.exists(path):
            return path
        return os.path.join(os.getcwd(), "viewer/index.html")


# Adapted from https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
class ExitListener:
    killed = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("Exiting...")
        self.killed = True


def is_collection(path):
    # Can either be a directory containing collection.json or the file itself.
    if (path.endswith("collection.json")) and os.path.isfile(path):
        return True
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "collection.json"))


def is_dataset(path):
    # Can either be a directory containing manifest.json or the file itself.
    if (path.endswith("manifest.json")) and os.path.isfile(path):
        return True
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "manifest.json"))


def serve_until_terminated(httpd: HTTPServer):
    # Poll for exit signal every second
    httpd.timeout = 1
    httpd.handle_timeout = lambda: None
    exit_listener = ExitListener()

    while not exit_listener.killed:
        try:
            httpd.handle_request()
        except Exception as e:
            print("Error occurred while handling request:", e)
    httpd.server_close()


def serve_tfe(port):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    httpd = HTTPServer(
        ("localhost", port),
        ReactAppRequestHandler,
    )
    serve_until_terminated(httpd)


def serve_directory(port):
    httpd = HTTPServer(
        ("localhost", port),
        CORSRequestHandler,
    )
    serve_until_terminated(httpd)


def main():
    parser = argparse.ArgumentParser(
        description="Opens a dataset in a local instance of Timelapse Feature Explorer (TFE)."
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to the dataset to be opened."
    )
    parser.add_argument(
        "--tfe_port",
        type=int,
        default=default_tfe_port,
        help="Port for the TFE server. Defaults to {}.".format(default_tfe_port),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_directory_port,
        help="Port for the dataset server. Defaults to {}.".format(
            default_directory_port
        ),
    )

    args = parser.parse_args()
    dataset_path = os.path.abspath(args.dataset_path)

    (tfe_port, tfe_reserved_socket), (directory_port, directory_reserved_socket) = (
        acquire_ports([args.tfe_port, args.port])
    )

    if tfe_port != args.tfe_port:
        print(f"Port {args.tfe_port} is in use. Using port {tfe_port} for TFE instead.")
    if directory_port != args.port:
        print(
            f"Port {args.port} is in use. Using port {directory_port} for the directory server instead."
        )

    # Change working directory to the provided dataset directory
    new_cwd = os.getcwd()
    if os.path.isfile(dataset_path):
        new_cwd = os.path.dirname(dataset_path)
        dataset_path = os.path.basename(dataset_path)
    elif os.path.isdir(dataset_path):
        new_cwd = dataset_path
        dataset_path = ""
    else:
        raise ValueError("The specified path does not exist: {}".format(dataset_path))
    os.chdir(new_cwd)

    url = "http://localhost:{}/viewer/viewer?".format(tfe_port)
    abs_path = os.path.abspath(dataset_path)
    if is_collection(abs_path):
        url += "collection=http://localhost:{}/{}".format(
            directory_port, os.path.relpath(dataset_path or "collection.json")
        )
    elif is_dataset(abs_path):
        url += "dataset=http://localhost:{}/{}".format(
            directory_port, os.path.relpath(dataset_path or "manifest.json")
        )
    else:
        raise ValueError(
            "Could not find a manifest.json or collection.json in the specified path: {}".format(
                os.path.abspath(dataset_path)
            )
        )

    # Start TFE server in a separate process
    tfe_reserved_socket.close()
    tfe_process = Process(target=serve_tfe, args=(tfe_port,))
    tfe_process.start()

    # Prevents a bug where the page fails to load because the TFE server hasn't
    # been initialized yet.
    sleep(1)
    print("Opening TFE at", url)
    webbrowser.open(url)

    # Blocks until the server kills the process.
    directory_reserved_socket.close()
    serve_directory(directory_port)

    # Exit gracefully
    tfe_process.terminate()
    tfe_process.join()


if __name__ == "__main__":
    main()
