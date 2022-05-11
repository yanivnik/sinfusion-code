import torch  #pylint: disable=unused-import
import csv
import datetime
import os
import socket
# import shutil
from subprocess import check_output
import shlex
import json
import itertools
from bisect import bisect


flatten = lambda x: list(itertools.chain.from_iterable(x))
two_tuple = lambda x: x if isinstance(x, (list, tuple)) else (x, x)

###############################################################################
#                   BLA
###############################################################################
def send_command(command):
    print(command)
    byte_output = check_output(shlex.split(command))
    out = byte_output.decode("utf-8")
    return out


def dump_obj_with_dict(obj, save_to_path):
    obj.dump_time = now()
    with open(save_to_path, 'w') as f:
        json.dump(obj.__dict__, f, indent=4, sort_keys=True, default=str)


def load_dict_to_obj(load_from_path):
    class A:
        pass
    args = A()
    with open(load_from_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args


###############################################################################
#                   TENSOR NORMALIZATION
###############################################################################
def log_with_zeros(x, zero_shift=1):
    """
    Return log of x
    Transform zeros: 0 --> min(log(non-zeros)) - zero_shift
    """
    true_log = x.log()
    bad_idxs = torch.isinf(true_log) | torch.isnan(true_log)
    minv = true_log[~bad_idxs].min()
    true_log[bad_idxs] = minv - zero_shift
    return true_log


###############################################################################
#                   LOGGING ETC.
###############################################################################
def now():
    return datetime.datetime.now()  # .strftime('%H:%d:%S')


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def server_name():
    hn = socket.gethostname()
    return hn[:hn.find('.')]


def base_log_string():
    return f"[{now()} {server_name()} ({os.environ['CUDA_VISIBLE_DEVICES']})]"


def save_weights(dirpath, model, epoch=None, batch=None, ext_text=None):
    weights_fname = 'weights'
    if epoch is not None:
        weights_fname += '-%d' % epoch
    if batch is not None:
        # weights_fname = 'weights-%d-%d-%s.pth' % (epoch, batch, ext_text)
        weights_fname += '-%d' % batch
    if ext_text is not None:
        weights_fname += '-%s' % ext_text
    weights_fname += '.pth'

    weights_fpath = os.path.join(dirpath, weights_fname)
    torch.save({
            'batch': batch,
            'epoch': epoch,
            'state_dict': model.state_dict()
        }, weights_fpath)
    print('saved weights to:', weights_fpath)


def load_weights(model, fpath, device='cuda'):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath, map_location=device)
    model.load_state_dict(weights['state_dict'])


def resume_model_from(weights_dir, model, epoch):
    weights_name = 'weights-{}.pth'.format(epoch)
    weights_path = os.path.join(weights_dir, 'weights', weights_name)
    load_weights(model, weights_path)


class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Scheduler:
    def __init__(self, start, end, n_epochs, sched_type='lin'):
        self.sched_type = sched_type
        if n_epochs is None:
            self.schedule = [torch.tensor(end)]
        else:
            high = torch.tensor(end)
            low = torch.tensor(start) if start is not None else None
            if sched_type == 'lin':
                self.schedule = torch.linspace(low, high, n_epochs)#.cuda()
            elif sched_type == 'log':
                self.schedule = torch.logspace(torch.log10(low.float()), torch.log10(high.float()), n_epochs)#.cuda()
            elif sched_type == 'milestones':
                self.values = end
                self.milestones = n_epochs
            else:
                raise Exception('Unknown scheduler type', sched_type)

    def get_current(self, t):
        if self.sched_type == 'milestones':
            return self.values[bisect(self.milestones, t)]

        if t < len(self.schedule):
            return self.schedule[t].item()
        else:
            return self.schedule[-1].item()


class Logger(object):
    """
    Usage:
    # initialize logger
    logger = Logger(logfilepath='/path/to/logfile.csv', cols=['colname_a', 'colname_b', ...])
    ...
    # log something
    logger.log({
                'colname_a': value_a,
                'colname_b': value_b,
               },
               verbose=True)
    """
    def __init__(self, logfilepath, cols):
        self.logfilepath = logfilepath
        self.cols = cols
        self.base_cols = ['date']

        log_dir = os.path.split(logfilepath)[0]
        os.makedirs(log_dir, exist_ok=True)
        self.add2csv(self.logfilepath, self.base_cols + self.cols)

    def add2csv(self, csvpath, fields):
        with open(csvpath, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def assert_row(self, row_dict):
        a = set(self.cols)
        b = set(row_dict.keys())
        if a != b:
            raise Exception('key: ', b.symmetric_difference(a), 'is missing or not expected')

    def log(self, row_dict, verbose=True):
        self.assert_row(row_dict)
        fields = [now()] + [row_dict[k] for k in self.cols]
        self.add2csv(self.logfilepath, fields)

        if verbose:
            base_str = '[{} {} ({})]'.format(now(), server_name(), os.environ['CUDA_VISIBLE_DEVICES'])
            print('logged', base_str, row_dict)
