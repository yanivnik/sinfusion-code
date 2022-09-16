import os
import jinja2
import common_utils as utils


def create_videos_html(base_dir, html_filename, vids, vid_size, column_max_width):
    """
    base_dir: path to folder, where the HTML file will be saved, and where all mp4 files are available under
    html_filename: e.g. if "bla" the result will be bla.html
    vid_size: size in pixels (e.g. 340)
    column_max_width: width per column in percentage (e.g. 15 means 15% of the total page width)
    vids: list of "sections" where each section is a list of [title, list-with-relative-paths-to-vids]

    ***************
    See Example:
    base_dir = '/home/nivh/data/projects_data/patch_diffusion'
    html_filename = 'bla.html'
    column_max_width = 15
    vid_size = 340
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
    vids = []
    for dname in dnames:
        l = [
            dname, [os.path.join('./gt.mp4')] + [f'./{dname}/{i}.mp4' for i in range(1, 5)]
        ]
        vids.append(l)

    """

    TEMPLATE_FILE = r"common_utils/html_videos_template.html"
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(
        vids=vids,
        vid_size=vid_size,
        column_max_width=column_max_width,
    )

    # save html to file
    output_path = os.path.join(base_dir, html_filename)
    with open(output_path, 'w') as f:
        f.write(outputText)
        print(utils.common.now(), 'saved to:', output_path)
