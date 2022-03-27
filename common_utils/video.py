import os
# import moviepy.editor as mpy
import cv2
import numpy as np
# % matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import animation
import torch
import imageio
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


def frames_to_gif(frames_dir, clip_path, fps, scale=1):
    start_frame = 0
    end_frame = sorted(map(int, [i[:-4] for i in os.listdir(frames_dir)]))[-1]
    #     fps = end_frame+1

    vid = main_utils.read_original_video(frames_dir, start_frame, end_frame, device='cpu', verbose=False)
    # print('vid_shape:', vid.shape)
    if scale != 1:
        vid = torch.nn.functional.interpolate(vid, size=[vid.shape[2], vid.shape[3] * scale, vid.shape[4] * scale])
    frames = [tensor2npimg(vid[:, :, t, :, :]) for t in range(vid.shape[2])]
    kwargs_write = {'fps': fps, 'quantizer': 'nq'}
    imageio.mimwrite(clip_path, frames, 'GIF-FI', **kwargs_write)


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


###############################################################################
#                   Old Stuff                                                 #
###############################################################################
# def extract_frames_from_video_cv2(video_path, frame_dir):
#     vidcap = cv2.VideoCapture(video_path)
#     success,image = vidcap.read()
#     count = 0
#     while success:
#       cv2.imwrite(os.path.join(frame_dir, "%d.png" % count), image)
#       success,image = vidcap.read()
#       print('Read a new frame: ', success)
#       count += 1
#     print(f'Extracted {count} Frames to:', frame_dir)


# def generate_gif(frames_folder, gif_path):
#     import imageio
#     kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
#     images = []  # output array
#     # for file in os.listdir(frames_folder):  # I think we need to sort the listdir in our case of videos
#     #     im = np.expand_dims(utils.ben_image.imread(frames_folder + file), 0)
#     #     im = torch.tensor(im).contiguous().permute(0, 3, 1, 2).detach().cuda()
#     #     images.append(im.detach().clone().squeeze(0).permute(1, 2, 0).cpu())
#     imageio.mimsave(gif_path, images, 'GIF-FI', **kwargs_write)


# def frames_to_mp4(frames_dir, start_frame, end_frame, clip_path, fps):
#     clip = mpy.ImageSequenceClip([os.path.join(frames_dir, f'{i}.png') for i in range(start_frame, end_frame+1)], fps=fps)
#     clip.write_videofile(clip_path, logger=None)

# def make_vid(frames_dir, vid_path, fps):
#     start_frame = 0
#     end_frame = sorted(map(int, [i[:-4] for i in os.listdir(frames_dir)]))[-1]
#     frames = [os.path.join(frames_dir, f'{i}.png') for i in range(start_frame, end_frame + 1)]
#     frames_to_mp4(frames, vid_path=vid_path, fps=fps)


# def frames_to_mp4(frames, vid_path, fps):
#     vid = mpy.ImageSequenceClip(frames, fps=fps)
#     vid.write_videofile(vid_path, logger=None)
# #
# def read_frames_from_dir(frames_dir, extend_to=None):
#     start_frame = 0
#     end_frame = sorted(map(int, [i[:-4] for i in os.listdir(frames_dir)]))[-1]
#     frames = [os.path.join(frames_dir, f'{i}.png') for i in range(start_frame, end_frame + 1)]
#     # if extend_to is None:
#     #     extend_to = end_frame
#     # frames += [os.path.join(frames_dir, f'{end_frame}.png') for _ in range(end_frame + 1, extend_to + 1)]
#     return frames, end_frame
# #
# def make_vid(frames_dir, vid_path, fps, extend_to=None):
#     frames, end_frame = read_frames_from_dir(frames_dir, extend_to)
#     frames_to_mp4(frames_dir, vid_path=vid_path, fps=fps)
#     return end_frame

# def frames_to_mp4_old(frames, vid_path, fps):
#     vid = mpy.ImageSequenceClip(frames, fps=fps)
#     vid.write_videofile(vid_path, logger=None)


# def make_vid(frames_dir, vid_path, fps, ext='png'):
#     start_frame = 0
#     end_frame = sorted(map(int, [i[:-4] for i in os.listdir(frames_dir)]))[-1]
#     frames = [os.path.join(frames_dir, f'{i}.{ext}') for i in range(start_frame, end_frame + 1)]
#     frames_to_mp4(frames, vid_path=vid_path, fps=fps)
#
#
# def save_as_clip(frames_dir, clip_path, format='mp4', fps=10):
#     start_frame = 0
#     end_frame = sorted(map(int, [i[:-4] for i in os.listdir(frames_dir)]))[-1]
#     frames = [os.path.join(frames_dir, f'{i}.png') for i in range(start_frame, end_frame + 1)]
#
#     if format == 'mp4':
#         import moviepy.editor as mpy
#         clip = mpy.ImageSequenceClip([os.path.join(frames_dir, f'{i}.png') for i in range(start_frame, end_frame+1)], fps=fps)
#         clip.write_videofile(clip_path)
#     if format == 'gif':
#         # import main_utils
#         # import common_utils as utils
#         import imageio
#         # vid = main_utils.read_original_video(frames_dir, start_frame, end_frame, device='cpu', verbose=False)
#         # frames = [utils.image.tensor2npimg(vid[:, :, t, :, :]) for t in range(vid.shape[2])]
#         kwargs_write = {'fps': fps, 'quantizer': 'nq'}
#         imageio.mimwrite(clip_path, frames, 'GIF-FI', **kwargs_write)
#
