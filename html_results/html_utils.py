import os
import jinja2
import common_utils as utils

# html_filename='niv.html'
# output_path = '/home/yanivni/data/tmp/organized-outputs/Video Generation/air_balloons/bla.html'
frame_rate = 10
base_dir = '/home/nivh/data/projects_data/patch_diffusion'
html_filename = 'bla.html'

dnames = [
    'negative-frame-diff-extrapolation',
    'every-N=5-frames-is-fd-hop-and-in-between-is-interpolated',
    'framediff-long-training-122-epochs',
    'partial-training-24-epochs-with-fe',
    'every-N=2-frames-is-fd-hop-and-in-between-is-interpolated',
    '0-1-2-3-4-and-0-5-frame-diff-hops',
    '5-frame-diff-embedding',
    'full-training-with-fe',
    'fe-long-training-130-epochs',
    'every-N=5-frames-is-fd-hop-and-in-between-is-interpolated-between-prev-and-hopped',
]

# vids = [
#     ['GT', (
#         os.path.join('./gt.mp4'),
#     )],
# ]
vids = []
for dname in dnames:
    l = [
        dname, [os.path.join('./gt.mp4')] + [f'./{dname}/{i}.mp4' for i in range(1, 5)]
    ]
    vids.append(l)

TEMPLATE_FILE = r"html_videos_template.html"
templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template(TEMPLATE_FILE)
outputText = template.render(
    vids=vids,
    frame_rate=frame_rate,
    column_max_width=15,
    vid_size=340,
)
# save html to file
output_path = os.path.join(base_dir, html_filename)
with open(output_path, 'w') as f:
    f.write(outputText)
    print(utils.common.now(), 'saved to:', output_path)
