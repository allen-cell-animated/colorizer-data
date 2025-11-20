#!/usr/bin/env python3

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from multiprocessing import Process
import argparse
import os
import re
import requests
import signal
import shutil
import socket
from time import sleep
from typing import List, Tuple
import webbrowser
import zipfile


"""
# Directory layout:

tfe_open.py       # This script
viewer/
    default/      # Built-in viewer files (committed to git)
    latest/       # User-modifiable viewer files, can be updated (gitignored).
                  # Will be initialized by copying from `default/` on first run.

"""

viewer_directory = "viewer"
viewer_directory_default = "default"
viewer_directory_latest = "latest"


def get_base_viewer_path():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(script_dir, viewer_directory)


def get_viewer_path():
    return os.path.join(get_base_viewer_path(), viewer_directory_latest)


def initialize_viewer_directory(force=False):
    """
    Copy files from the `viewer/default` to `viewer/latest` if `latest` doesn't
    exist. `latest` is set to be ignored by git, so users can modify it without
    affecting the repo.
    """
    latest_viewer_path = get_viewer_path()
    default_viewer_path = os.path.join(get_base_viewer_path(), viewer_directory_default)

    path_exists = os.path.exists(latest_viewer_path)
    has_html = os.path.exists(os.path.join(latest_viewer_path, "index.html"))
    if (not path_exists) or (not has_html) or force:
        shutil.rmtree(latest_viewer_path, ignore_errors=True)
        shutil.copytree(default_viewer_path, latest_viewer_path)


def fetch_latest_viewer_info() -> tuple[str, str] | None:
    """
    Returns a tuple of the latest version string and the download URL for the
    latest release, or None if the request fails.
    """
    url = "https://api.github.com/repos/allen-cell-animated/timelapse-colorizer/releases/latest"
    url_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(url, headers=url_headers, timeout=1)
    if response.status_code != 200:
        return None
    release_info = response.json()
    tag_name = release_info["tag_name"][1:]  # Strip leading 'v'
    download_url = release_info["assets"][0]["browser_download_url"]
    return tag_name, download_url


def update_to_latest_viewer(download_url: str):
    print("\nUpdating to the latest version...")
    viewer_path = get_viewer_path()

    response = requests.get(download_url, timeout=10)
    if response.status_code != 200:
        print("Warning: Could not download the latest TFE version.")
        return

    try:
        # Clear existing viewer files
        if os.path.exists(viewer_path):
            shutil.rmtree(viewer_path)
        os.makedirs(viewer_path, exist_ok=True)

        # Write and extract the zip file.
        # Files in the zip are contained in a top-level "viewer/" folder.
        zip_path = os.path.join(viewer_path, "tfe_latest.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)
        parent_path = os.path.dirname(viewer_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(parent_path)

        # Move files from the extracted folder to viewer_path and cleanup
        extracted_folder = os.path.join(parent_path, "viewer")
        shutil.copytree(extracted_folder, viewer_path, dirs_exist_ok=True)
        os.remove(zip_path)
        shutil.rmtree(extracted_folder)
        print("Update complete.")
    except Exception as e:
        print("Error: Could not install the latest TFE version:", e)
        print("Restoring the built-in version.")
        initialize_viewer_directory(force=True)


def get_version_from_html(html_content: str) -> str | None:
    # Find meta tag: <meta name="version" content="x.x.x">
    match = re.search(r'<meta\s+name="version"\s+content="([^"]+)"', html_content)
    if match:
        return match.group(1)
    return None


def get_current_viewer_version() -> str | None:
    """
    Gets the current version of the viewer from the built viewer file, as a
    "X.Y.Z" semver string.
    """
    index_html_path = os.path.join(get_viewer_path(), "index.html")
    with open(index_html_path, "r") as f:
        html_content = f.read()
        return get_version_from_html(html_content)


def check_for_and_update_tfe(allow_update: bool) -> None:
    initialize_viewer_directory()
    # Check for TFE version updates
    current_version = None
    latest_version_info = None
    try:
        current_version = get_current_viewer_version()
    except Exception as e:
        print(
            "Warning: Could not get current TFE version. Files may be corrupted or missing.",
            e,
        )
    try:
        latest_version_info = fetch_latest_viewer_info()
        if latest_version_info is None:
            print("Warning: Could not fetch latest TFE version.")
        else:
            if current_version != latest_version_info[0]:
                print(
                    "A new version of TFE is available! (new: {}, current: {})".format(
                        latest_version_info[0], current_version
                    )
                )
                if allow_update:
                    update_to_latest_viewer(latest_version_info[1])
                else:
                    print(
                        "Run this script with the '--latest' flag to automatically update to the latest version."
                    )
            else:
                print("TFE is up to date. (version {})".format(current_version))
    except Exception as e:
        print("Warning: Could not fetch latest TFE version:", e)
    print()


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


# Adapted from https://stackoverflow.com/a/21957017 and
# https://gist.github.com/dustingetz/5348582.
class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS for all domains
        self.send_header("Access-Control-Allow-Origin", "*")
        # Request browser not to cache files
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        SimpleHTTPRequestHandler.end_headers(self)


class ReactAppRequestHandler(CORSRequestHandler):
    def translate_path(self, path):
        # Serve the React app's index.html for any path that doesn't map to a
        # file for client-side routing.
        path = super().translate_path(path)
        if os.path.exists(path):
            return path
        return os.path.join(os.getcwd(), "index.html")


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


def serve_until_terminated(httpd: ThreadingHTTPServer):
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
    os.chdir(get_viewer_path())

    httpd = ThreadingHTTPServer(
        ("localhost", port),
        ReactAppRequestHandler,
    )
    serve_until_terminated(httpd)


def serve_directory(port):
    httpd = ThreadingHTTPServer(
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
    parser.add_argument(
        "--latest",
        action="store_true",
        default=False,
        help="Whether to automatically upgrade to the latest version of TFE.",
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

    check_for_and_update_tfe(args.latest)

    url = "http://localhost:{}/viewer?".format(tfe_port)
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
    print("Press Ctrl+C to exit.\n")
    webbrowser.open(url)

    # Blocks until the server kills the process.
    directory_reserved_socket.close()
    serve_directory(directory_port)

    # Exit gracefully
    tfe_process.terminate()
    tfe_process.join()


if __name__ == "__main__":
    main()
