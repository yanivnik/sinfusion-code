import sys
import common_utils as utils
from jobs_utils import shipping


def main(wandb_sweep_id):
    # build argss (job dicts)
    argss = [{}]
    hosts = ['n103']
    forbidden_gpus = {
        # ('n114', 'mcluster11.wisdom.weizmann.ac.il'): [7],
        # 'n110': [6, 7],
        # 'n108': [0],
    }

    jobs_folder = '/home/yanivni/data/projects_data/jobs/single-image-diffusion'
    run_folder = '/home/yanivni/data/remote_projects/single-image-diffusion'
    script_name = f'wandb agent weizmann/single-image-diffusion/{wandb_sweep_id}'
    n_jobs = shipping.ship_as_queue(argss, hosts, script_name, jobs_folder, run_folder,
                                    gpus_per_job=1, waiting_time=15, max_jobs_on_host=50, forbidden_gpus=forbidden_gpus)
    print(f'SENT ALL! (TOTAL {n_jobs})')
    print(utils.common.now())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python send_many_wandb.py <WANDB_SWEEP_ID>')
        exit()
    main(sys.argv[1])
