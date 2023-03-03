import os

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import animation
# % matplotlib inline
from matplotlib import pyplot as plt

from .image import imread
from .common import send_command
from .image import tensor2npimg


###############################################################################
#              Read/Write video-frames                                        #
###############################################################################
def frames_to_mp4(frames_dir, vid_path, fps, fns=None):
    if fns is None:
        fns = sorted(map(int, [i[:-4] for i in os.listdir(frames_dir)]))
        # frames = [os.path.join(frames_dir, f'{i}.png') for i in range(fns[0], fns[1] + 1)]

    frames = [os.path.join(frames_dir, f'{i}.png') for i in fns]

    frame = cv2.imread(frames[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(cv2.imread(frame))
    cv2.destroyAllWindows()
    video.release()


def torchvid2mp4(vid, path, fps=10):
    """ vid is CTHW """
    torchvision.io.write_video(path, tensor2npimg(vid[None, ...], to_numpy=False).permute(1, 2, 3, 0), fps=fps)


def read_frames_from_dir(path):
    """
    Read frames from a directory, which includes frames of a video starting from 1.png to N.png
    """
    frames = []
    for frame in range(1, len(os.listdir(path)) + 1):
        frames.append(imread(os.path.join(path, f'{frame}.png')))
    return torch.cat(frames, dim=0)

###############################################################################
#               FFmpeg utils                                                  #
###############################################################################
def decrease_video_frame_rate(src_path, dst_path, ratio=1.0):
    """from: https://stackoverflow.com/a/45465730/736211 """
    command = """ffmpeg -y -i %s -vf "setpts=%.2f*PTS" -r 24 %s""" % (
        src_path,
        ratio,
        dst_path
    )
    send_command(command)


def download_youtube_video(url, start="000000.00", num_seconds="00", video_path=None, max_height=1080):
    """ from: https://unix.stackexchange.com/a/282413/403616 """
    assert video_path is not None

    # get full url of the video
    command = """youtube-dl -f bestvideo[height<=%d] --youtube-skip-dash-manifest -g %s""" % (
        max_height,
        url,
    )
    out = send_command(command)
    video_url = out.split('\n')[0]

    # download the relevant part using ffmpeg
    command = """ffmpeg -ss %s:%s:%s -i "%s" -t 00:00:%s.00 -c copy -y %s""" % (
        start[:2], start[2:4], start[4:6],#, start[7:],
        video_url,
        num_seconds,
        video_path,
    )
    send_command(command)
    print('saved video to path:', video_path)


def get_number_of_frames(video_path):
    """from: https://stackoverflow.com/a/28376817/736211 """
    command = """ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 %s""" % (
        video_path,
    )
    n_frames = send_command(command)
    return int(n_frames.strip())


def extract_frames_from_video(video_path, frame_dir, frame_file_glob=r"%001d.png", makedir=True, verbose=True):
    if makedir and not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=True)

    command = """ffmpeg -i %s -vf mpdecimate,setpts=N/FRAME_RATE/TB %s""" % (
        video_path,
        frame_dir + "/" + frame_file_glob,
    )
    print("extracting frames from video")
    send_command(command)

    if verbose:
        print(f'extracted {len(os.listdir(frame_dir))} frames')


def video_from_frame_directory(frame_dir, video_path, frame_file_glob=r"%001d.png", framerate=24, with_text=False, use_mpeg4=True):
    """Build a mp4 video from a directory frames
        note: crop_to_720p crops the top of 1280x736 images to get them to 1280x720
    """
    vf_arg = r"drawtext=fontfile=/usr/share/fonts/dejavu/DejaVuSans.ttf: text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5"
    command = r"""ffmpeg -framerate %s -f image2 -i %s -q:v 1 %s %s %s""" % (
        framerate,
        frame_dir + "/" + frame_file_glob,
        "-vcodec mpeg4" if use_mpeg4 else "",
        '-vf "%s"' % vf_arg if with_text else "",
        video_path,
    )
    print("building video from frames")
    send_command(command)


###############################################################################
#                   HTML Video (for notebooks)                                #
###############################################################################
def html_vid(vid, interval=100):
    """
        Use in jupyter:
        anim = html_vid(q_vid)
        HTML(anim.to_html5_video())
    """
    video = vid.detach().cpu().numpy()[0]
    video = np.transpose(video, (1, 2, 3, 0))
    video = (video + 1) / 2
    video = np.clip(video, 0, 1)
    fig = plt.figure()
    im = plt.imshow(video[0, :, :, :])
    plt.axis('off')
    plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(video[0, :, :, :])

    def animate(i):
        im.set_data(video[i, :, :, :])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0], interval=interval)
    return anim