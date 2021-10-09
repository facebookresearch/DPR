#!/usr/bin/env python
from typing import Dict

import argparse
import os
import subprocess

import enum
import glob
import turibolt as bolt


class Cluster(enum.Enum):
    SIMCLOUD = "SIMCLOUD"
    APC = "APC"


def get_cluster(task: bolt.Task) -> Cluster:
    return Cluster(task.cluster_type)


def get_remote_dir(cluster: Cluster) -> str:
    """Returns the copy destination for the specified cluster.

    The default "working directory" is different for Simcloud and APC, so this lets us make sure
    our local files get copied to the correct location.
    """
    if cluster == Cluster.SIMCLOUD:
        return "/task_runtime/"
    elif cluster == Cluster.APC:
        return "/mnt/task_runtime/"
    else:
        raise ValueError(f"Cluster {cluster} is not supported.")


def get_upload_command(command_args: Dict[str, str], task: bolt.Task) -> str:
    """Returns the rsync command that copies your local code to the task host.

    The command differs depending on cluster since, for example, APC requires the use of
    an SSH proxy. To find out what kind of SSH config is need for a specific cluster, run
    `bolt task show <task_id>`.
    """
    cluster = get_cluster(task)
    if cluster == Cluster.SIMCLOUD:
        return (
            'rsync -Pauv -e "ssh -i {bolt_config_dir}/bolt_ssh_key '
            '-p {port}" --exclude=".git/" '
            '--exclude-from="$(git -C {local_dir} ls-files --exclude-standard -oi --directory > /tmp/excludes; echo /tmp/excludes)" '
            "{local_dir}/* root@{ip}:{remote_dir}"
        ).format(**command_args)
    elif cluster == Cluster.APC:

        available_ssh_config_list = glob.glob(
            "%s/.turibolt/ssh_config.*" % (os.path.expanduser("~"))
        )

        newest_ssh_config = sorted(
            available_ssh_config_list, key=_ssh_config_version
        )[-1]
        cluster_name = task.resources.cluster.replace("apc_", "")
        return (
            'rsync -Pauv -e "ssh -i {bolt_config_dir}/bolt_ssh_key '
            f"-F {newest_ssh_config} "
            f"-o 'ProxyCommand=bolt_tunnel --proxy proxy-{cluster_name}-bolt.apple.com:443 --dest %h:%p "
            "--task_id {task_id}' "
            '-p {port}" --exclude=".git/" '
            '--exclude-from="$(git -C {local_dir} ls-files --exclude-standard -oi --directory > /tmp/excludes; echo /tmp/excludes)" '
            "{local_dir}/* root@{ip}:{remote_dir}"
        ).format(**command_args)
    else:
        raise ValueError(f"Cluster {cluster} is not supported.")


def _ssh_config_version(ssh_config_name):
    version_str = ssh_config_name.rsplit(".", maxsplit=1)[-1]
    if version_str[0] != "v":
        raise ValueError(
            "Invalid SSH config name: {}".format(ssh_config_name)
        )
    return int(version_str[1:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync local files to a machine launched by a Bolt task. "
        "This will sync local files and then watch the "
        "filesystem for changes, re-uploading changes as they are made. "
        "Files matching patterns in a local .gitignore path will be ignored. "
        "If you have AnyBar installed, this will add a new dot to your "
        "menu bar that is orange when a sync is in progress and green "
        "otherwise. If you want to run multiple at once, choose different"
        "AnyBar ports with the ANYBAR_PORT variable."
    )
    parser.add_argument("id", help="Bolt task id")
    parser.add_argument("--local-dir", default=".", help="Directory to sync")

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run sync once and stop. " "If set, the usual upload will complete and stop.",
    )
    args = parser.parse_args()

    task = bolt.get_task(args.id)
    cluster = get_cluster(task)

    command_args = {
        "port": task.opened_ports["SSH_PORT"],
        "ip": task.host_ip_address,
        "task_id": task.id,
        "local_dir": os.path.realpath(args.local_dir),
        "remote_dir": get_remote_dir(cluster),
        "bolt_config_dir": os.environ.get("BOLT_CONFIG_DIR", "$HOME/.turibolt"),
    }
    upload_command = get_upload_command(command_args, task)

    print("Watching for changes. Running:\n" + upload_command)
    subprocess.call(upload_command, shell=True)

    # Start watching
    if not args.once:
        anybar_bin = "/Applications/AnyBar.app/Contents/MacOS/AnyBar"
        use_anybar = os.path.exists(anybar_bin)

        if use_anybar:
            # Start AnyBar. This will start listening on the port specified by
            # $ANYBAR_PORT.
            subprocess.Popen("/Applications/AnyBar.app/Contents/MacOS/AnyBar")

            anybar_port = os.getenv("ANYBAR_PORT", "1738")
            anybar_command = ("printf {{color}} | " "nc -4u -w0 localhost {port}").format(
                port=anybar_port
            )
            start_command = anybar_command.format(color="orange")
            done_command = anybar_command.format(color="green")

            upload_command = ("{start_command}; {upload_command} && " "{done_command};").format(
                start_command=start_command,
                upload_command=upload_command,
                done_command=done_command,
            )
        try:
            # Exclude .git because its mtime gets updated every time
            # `git status` is run, which can be quite often when you use
            # fancy shell prompts.
            #
            # Also, escape the double-quotes in the upload command since the whole thing is being
            # shoved inside a bash -c "" invocation.
            watch_command = (
                'fswatch -or0 -l 0.2 --exclude ".\*.git.\*" {local_dir} | '
                'xargs -0 -n 1 bash -c "{upload_command}";'
            ).format(local_dir=args.local_dir, upload_command=upload_command.replace('"', '\\"'))
            print(watch_command)
            subprocess.Popen(watch_command, shell=True).wait()
        except:
            # End quietly.
            pass
