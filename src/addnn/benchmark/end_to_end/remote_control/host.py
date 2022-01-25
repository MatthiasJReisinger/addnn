import ipaddress
import subprocess
from typing import Optional


def execute_remote_command(host: str, user: str, command: str, ssh_key_path: Optional[str] = None) -> str:
    ssh_command = ["ssh"]
    if ssh_key_path is not None:
        ssh_command += ["-i", ssh_key_path]
    ssh_command += ["{}@{}".format(user, host), command]
    process = subprocess.run(ssh_command, text=True)
    if process.returncode != 0:
        raise Exception("Could not execute command '{}' on {}, returncode={}, stdout='{!r}', stderr='{!r}'".format(
            command, host, process.returncode, process.stdout, process.stderr))
    return process.stdout


def copy_from_remote_host(host: str,
                          user: str,
                          source_path: str,
                          target_path: str,
                          is_directory: bool = False,
                          ssh_key_path: Optional[str] = None) -> str:
    host_ip_version = ipaddress.ip_address(host).version

    if host_ip_version == 6:
        formatted_host_ip = "[{}]".format(host)
    else:
        formatted_host_ip = host

    scp_command = ["scp"]
    if ssh_key_path is not None:
        scp_command += ["-i", ssh_key_path]
    if is_directory:
        scp_command += ["-r"]
    scp_command += ["{}@{}:{}".format(user, formatted_host_ip, source_path)]
    scp_command += [target_path]
    process = subprocess.run(scp_command, text=True)
    if process.returncode != 0:
        raise Exception("Could not copy file '{}' from {}, returncode={}, stdout='{!r}', stderr='{!r}'".format(
            source_path, host, process.returncode, process.stdout, process.stderr))
    return process.stdout
