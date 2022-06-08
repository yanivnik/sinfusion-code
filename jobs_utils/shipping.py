__author__ = "Niv Haim (niv.haim@weizmann.ac.il)"

import os
import subprocess
# import shlex
import datetime
import dateutil.tz
import numpy as np
import time
from .gpu_avail import list_gpus
from collections import defaultdict


# jobfile_template = """
# #!/usr/bin/env bash
#
# export QT_QPA_PLATFORM='offscreen'
# . "${{HOME}}/miniconda3/etc/profile.d/conda.sh"
# conda activate nivh-21.12
# which python
# cd {run_folder}
# pwd
#
# echo 'launching to host {host}...'
# CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 nohup python {script_name} {args} 1>{outfile} 2>{errfile} &
#
# echo 'exiting...'
# exit
# """


jobfile_template = """
#!/usr/bin/env bash

export QT_QPA_PLATFORM='offscreen'
. "${{HOME}}/miniconda3/etc/profile.d/conda.sh"
conda activate base
which python
cd {run_folder}
pwd

echo 'launching to host {host}...'
CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 nohup {script_name} {args} 1>{outfile} 2>{errfile} &

echo 'exiting...'
exit
"""


def dict2str(d):
    s = ''
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                s += f'--{k}'
        elif isinstance(v, (list, tuple)):
            s += f'--{k}={" ".join(map(str, v))}'
        elif v is None:
            continue
        else:
            s += f'--{k}={v}'
        s += ' '
    return s


def send_job(host, gpu_id, script_name, args, jobs_folder, run_folder):
    # create temporary jobfile
    time_str = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')
    rand_suffix = '%d' % np.random.randint(1e5,1e6)
    job_folder = os.path.join(jobs_folder, f'{time_str}_{rand_suffix}')
    os.makedirs(job_folder, exist_ok=True)
    runfile_path = os.path.join(job_folder, f'{time_str}_{rand_suffix}.sh')

    outfile_path = os.path.join(job_folder, f'{time_str}_{rand_suffix}.out')
    errfile_path = os.path.join(job_folder, f'{time_str}_{rand_suffix}.err')

    args_str = dict2str(args)
    with open(runfile_path, 'w') as f:
        f.write(jobfile_template.format(
            **{
                'run_folder': run_folder,
                'host': host,
                'gpu_id': gpu_id,
                'script_name': script_name,
                'args': args_str,
                'outfile': outfile_path,
                'errfile': errfile_path,
            }
        ))
    os.chmod(runfile_path, 0o755)

    # ship the job
    if isinstance(host, tuple):
        host, proxy = host
        cmd_list = ["ssh", '-o', f"ProxyCommand=ssh -W %h:%p {proxy}", host, runfile_path]
    else:
        cmd_list = ["ssh", host, runfile_path]
    print(subprocess.list2cmdline(cmd_list))
    ssh = subprocess.Popen(cmd_list,
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    result = ssh.stdout.readlines()
    print('command sent:')
    print(f'CUDA_VISIBLE_DEVICES={gpu_id} PYTHONUNBUFFERED=1 nohup {script_name} {args_str}')
    print('to view results:')
    print(f'tail -f {outfile_path}')
    print('to view errors:')
    print(f'tail -f {errfile_path}')
    return result


def get_all_available_gpus(hosts=None, forbidden_gpus=None):
    """
    Returns a list of available gpus (available == 0 used memory on that gpu)
    in the form (hostname, gpu_id)
    for example: [('n108', 2), ..., ('n110', 6)]
    """
    ben_gpu_map = list_gpus(hosts, max_workers=32)
    forbidden_gpus = forbidden_gpus if forbidden_gpus is not None else {}

    d = defaultdict(list)
    for host, gpus in ben_gpu_map.items():
        forbidden = forbidden_gpus.get(host, None)
        for gpu_id in range(len(gpus)):
            if gpus[gpu_id] == '-':
                if forbidden is not None and gpu_id in forbidden:
                    continue
                d[host].append(gpu_id)
    avs = []
    for host, gpu_ids in sorted(list(d.items()), key=lambda x: -len(x[1])):
        for gpu_id in gpu_ids:
            avs.append((host, gpu_id))
    # print('\n')
    # print('*'*100)
    # print(avs)
    # print('\n')
    return avs


class GPUManager:
    def __init__(self, hosts, max_jobs_on_host=50, forbidden_gpus=None):
        self.available_gpus = iter(get_all_available_gpus(hosts, forbidden_gpus))
        self.forbidden_gpus = forbidden_gpus if forbidden_gpus is not None else {}
        self.hosts_counts = {}
        self.max_jobs_on_host = max_jobs_on_host

    def get_gpus(self, num_gpus=1):
        gpus = []
        host = None
        good = num_gpus
        while good > 0:
            cur_host, gpu = next(self.available_gpus)
            if host is None:
                host = cur_host
            else:
                if cur_host != host:
                    raise StopIteration('no more available gpus')
            # forbidden_gpus = self.forbidden_gpus.get(cur_host, None)
            # if forbidden_gpus is not None and gpu in forbidden_gpus:
            #     continue
            # else:
            gpus.append(gpu)
            good -= 1
            # else:
            #     print(f'host: {cur_host}. unavailable gpu is free: {gpu}. available gpus: {available_gpus}')
        return host, gpus


def predicate_enough_available_gpus(ags, num_gpus):
    if len(ags) < num_gpus:
        return False
    host = ags[0][0]
    return set([ag[0] for ag in ags[:num_gpus]]).pop() == host


def get_gpu_or_wait(gpu_manager, hosts, num_gpus, waiting_time=15, max_jobs_on_host=50, forbidden_gpus=None):
    try:
        host, gpus = gpu_manager.get_gpus(num_gpus)
    except StopIteration:
        print('WAITING FOR GPU1')
        time.sleep(waiting_time)
        ags = get_all_available_gpus(hosts, forbidden_gpus)
        while not predicate_enough_available_gpus(ags, num_gpus):
            print('WAITING FOR GPU2')
            time.sleep(waiting_time)
            ags = get_all_available_gpus(hosts, forbidden_gpus)
        print(len(ags), 'AVAILABLE GPUS:', ags)
        gpu_manager = GPUManager(hosts, max_jobs_on_host=max_jobs_on_host, forbidden_gpus=forbidden_gpus)
        host, gpus = gpu_manager.get_gpus(num_gpus)
    return gpu_manager, host, gpus


def ship_as_queue(argss, hosts, script_name, jobs_folder, run_folder, gpus_per_job=1, waiting_time=15, max_jobs_on_host=50, forbidden_gpus=None):
    gpu_manager = GPUManager(hosts, max_jobs_on_host=max_jobs_on_host, forbidden_gpus=forbidden_gpus)
    print('gpus_per_job:', gpus_per_job)
    job_counter = 0
    n_jobs = len(argss)
    while argss:
        args = argss.pop()
        gpu_manager, host, gpus = get_gpu_or_wait(gpu_manager, hosts, gpus_per_job, forbidden_gpus=forbidden_gpus, waiting_time=waiting_time, max_jobs_on_host=max_jobs_on_host)
        gpu_id = ",".join(map(str, gpus))

        print('='*120)
        print(f'{job_counter+1} / {n_jobs} SENDING TO: {host} {gpu_id}')
        result = send_job(host, gpu_id, script_name, args, jobs_folder, run_folder)
        print(result)
        job_counter += 1

    return job_counter


