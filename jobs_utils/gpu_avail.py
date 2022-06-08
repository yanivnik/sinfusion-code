#!/net/mraid11/export/vision/.conda/env/gpustat/bin/python

__author__ = "Ben Feinstein (ben.feinstein@weizmann.ac.il)"

from typing import Iterable

import shlex
import subprocess
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from subprocess import PIPE, CalledProcessError


NVIDIA_SMI = 'nvidia-smi'
SSH_NO_SECURE = ['-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null']
Field = namedtuple('Field', 'name query type')


class NVSMIEntry:
    cmd = ''
    fields = ()

    def __init__(self, *args):
        self.data = {field.name: field.type(arg) for field, arg in zip(self.fields, args)}

    def __repr__(self) -> str:
        return "{cls}({fields})".format(
            cls=self.__class__.__name__,
            fields=', '.join(f'{k}={v}' for k, v in self.data.items()))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @classmethod
    def get_cmd(cls):
        return [
            'nvidia-smi',
            '--query-{cmd}={fields}'.format(
                cmd=cls.cmd,
                fields=','.join(field.query for field in cls.fields)),
            '--format=csv,noheader,nounits'
        ]


class GPUEntry(NVSMIEntry):
    cmd = 'gpu'
    fields = (
        Field('uuid', 'gpu_uuid', str),
        Field('bus', 'pci.bus', str),
        Field('name', 'name', str),
        Field('mem.total', 'memory.total', int),
        Field('mem.used', 'memory.used', int),
        Field('mem.free', 'memory.free', int),
        Field('util.gpu', 'utilization.gpu', int),
        Field('util.mem', 'utilization.memory', int),
    )


class PIDEntry(NVSMIEntry):
    cmd = 'compute-apps'
    fields = (
        Field('uuid', 'gpu_uuid', str),
        Field('pid', 'pid', int),
        Field('name', 'name', str),
        Field('mem.used', 'used_memory', int),
    )


def get_ssh_command(host, user=None, no_secure=True, proxy=None):
    if user is not None:
        host = f'{user}:{host}'
    return ['ssh',
            *(SSH_NO_SECURE if no_secure else []),
            *(['-o', f"ProxyCommand=ssh -W %h:%p {proxy}"] if proxy is not None else []),
            '-q',
            host]


def run_command(command, args=()):
    if isinstance(command, str):
        command = [command]
    command = command + list(args)
    # print(shlex.join(command))
    return subprocess.run(command, stdout=PIPE, stderr=PIPE,
                          check=True, text=True)


def query_gpus(command=()):
    result = run_command(command, GPUEntry.get_cmd()).stdout
    return parse_gpus(result)


def parse_gpus(result):
    result = result.strip()
    if not result:
        return {}
    gpus = [GPUEntry(*entry.split(', ')) for entry in result.split('\n')]
    gpus_by_uuid = {gpu['uuid']: gpu for gpu in gpus}
    return gpus_by_uuid


def query_compute_apps(command=()):
    result = run_command(command, PIDEntry.get_cmd()).stdout
    return parse_compute_apps(result)


def parse_compute_apps(result):
    result = result.strip()
    if not result:
        return {}
    pids = [PIDEntry(*entry.split(', ')) for entry in result.split('\n')]
    pids_by_uuid = defaultdict(list)
    for pid in pids:
        pids_by_uuid[pid['uuid']].append(pid)
    return pids_by_uuid


def query_all(command=(), sep='--SEPARATOR--'):
    commands = [shlex.join(GPUEntry.get_cmd()) + ';',
                shlex.join(['echo', sep]) + ';',
                shlex.join(PIDEntry.get_cmd())]
    # print(commands)
    result = run_command(command, commands).stdout
    # print(result)
    gpu_result, pid_result = result.split(sep, 1)
    gpus_by_uuid = parse_gpus(gpu_result)
    pids_by_uuid = parse_compute_apps(pid_result)
    for uuid in gpus_by_uuid:
        gpus_by_uuid[uuid]['pids'] = pids_by_uuid.get(uuid, [])
    gpus_list = sorted(gpus_by_uuid.values(), key=lambda x: x['bus'])
    return gpus_list


def remote_query_all(host=None, user=None):
    if host is not None:
        if isinstance(host, tuple):
            host, proxy = host
        else:
            proxy = None
        command = get_ssh_command(host=host, user=user, proxy=proxy)
    else:
        command = ()
    gpus = query_all(command)
    return gpus


def gpu_status(gpu):
    FREE = '-'
    ERR = 'x'
    NOT_FOUND = '[Not Found]'
    UNUSED_MEM_MAX = 11
    if any(p['name'] == NOT_FOUND for p in gpu['pids']):
        # for p in gpu['pids']:
        #     if p['name'] == NOT_FOUND:
        #         return '%s:%d' % (ERR, p['pid'])
        return ERR
    elif len(gpu['pids']) == 0 and gpu['mem.used'] > UNUSED_MEM_MAX:
        return ERR
    elif gpu['pids']:
        return gpu['util.gpu']
    return FREE

def list_gpus(hosts, max_workers=32):
    HOST_FMT = '{:^27}'
    ENTRY_FMT = '{:^3}'
    COM_ERR = 'COM'
    SEP=' | '
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for host in hosts:
            results[executor.submit(remote_query_all, host=host)] = host

    entries = {host: [] for host in hosts}
    for r in as_completed(results):
        try:
            # entries[results[r]] = [g['util.gpu'] if g['pids'] else FREE for g in r.result()]  # noqa
            entries[results[r]] = [gpu_status(g) for g in r.result()]
        except subprocess.CalledProcessError:
            entries[results[r]] = [COM_ERR]

    # n_gpus = max(len(e) for e in entries.values())
    # print('','-' *  len(HOST_FMT.format('')), *('-' * len(ENTRY_FMT.format('')) for i in range(n_gpus)), '', sep=SEP.replace('|', '+'))
    # print('', HOST_FMT.format('HOST'), *(ENTRY_FMT.format(str(i)) for i in range(n_gpus)), '', sep=SEP) # noqa
    # print('','-' *  len(HOST_FMT.format('')), *('-' * len(ENTRY_FMT.format('')) for i in range(n_gpus)), '', sep=SEP.replace('|', '+'))
    # for host, entry in entries.items():
    #     print('', HOST_FMT.format(host), *(ENTRY_FMT.format(str(e)) for e in entry), *(ENTRY_FMT.format('') for _ in range(n_gpus - len(entry))), '', sep=SEP)
    # print('','-' *  len(HOST_FMT.format('')), *('-' * len(ENTRY_FMT.format('')) for i in range(n_gpus)), '', sep=SEP.replace('|', '+'))
    return entries


# def main():
#     hosts = [
#         'n99.mcl.weizmann.ac.il',
#         'n100.mcl.weizmann.ac.il',
#         'n104.mcl.weizmann.ac.il',
#         'n105.mcl.weizmann.ac.il',
#         'n106.mcl.weizmann.ac.il',
#         'n107.mcl.weizmann.ac.il',
#         'n108.mcl.weizmann.ac.il',
#         'n109.mcl.weizmann.ac.il',
#         'n110.mcl.weizmann.ac.il',
#     ]
#     list_gpus(hosts, max_workers=32)


# if __name__ == "__main__":
#     # import argparse
#     # from pprint import pprint
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--host', type=str, default=None)
#     # args = parser.parse_args()
#     # pprint(remote_query_all(host=args.host))
#     main()
