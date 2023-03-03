

# SinFusion - Training Diffusion Models on a Single Image or Video
<a href=https://yanivnik.github.io/sinfusion> Project </a> | <a href=https://arxiv.org/abs/2211.11743> Arxiv </a></div>

Official PyTorch implementation of *"SinFusion - Training Diffusion Models on a Single Image or Video"*. 


### Data

You can download videos from this [Dropbox Videos Folder](https://www.dropbox.com/sh/75z6elt62i75kq4/AACFHqvo1L69zyY-JmhWO8eRa?dl=0) into ```./images``` folder.

Note that a video is represented as a directory with PNG files in the format \<frame number\>.png,
and should be placed within the ```./images/video``` folder.
For example:
```
some/path/sinfusion/images/video/my_video/
   1.png
   2.png
   3.png
   ...
```

### Training

Training a single-image DDPM:
```
python main.py --task='image' --image_name='balloons.png' --run_name='train_balloons_0'
```

Training a single-video DDPM (Predictor and Projector) for generation and extrapolation:
```
python main.py --task='video' --image_name='tornado' --run_name='tornado_video_model_0'
```

Training a single-video DDPM (Interpolator and Projector) for temporal upsampling:
```
python main.py --task='video_interp' --image_name='hula_hoop'  --run_name='hula_hoop_interpolate_0'
```

To choose the image/video, place the image/video in the data folder (see [Data](#data))
and replace the ```image_name``` argument with the name of the image/video.

Also, notice the ```run_name``` argument. This is the name of the experiment, and will be used in the generation to 
load the model. If this argument isn't specified, a default name will be used.

You may also change any configuration parameter via the command line. For a full list of configuration options, please see ```config.py```.

#### Async Training
To run the training script asynchronously, you can run a line similar to the following - 
```bash
PYTHONUNBUFFERED=1 nohup python main.py --task='video' --image_name='tornado' --run_name='tornado_video_model_0' > tornado_video_model_0.out &
```

### Generation
Most arguments are the same for the generation process as the training process.
Notice that the ```run_name``` argument should be the same as the one used in the training cmdline.

To generate a diverse set of images from a single-image DDPM:
```
python sample.py --task='image' --image_name='balloons.png' --run_name='train_balloons_0' [--sample_count=8] [--sample_size='(H, W)']
```

To generate a diverse video from a single-video DDPM:
```
python sample.py --task='video' --image_name='tornado' --output_video_len=100 --run_name=tornado_video_0
```

To extrapolate a video from a given frame using a single-video DDPM:
```
python sample.py --task='video' --image_name=tornado --output_video_len=100 --start_frame_index=33 --run_name=tornado_video_0
```

To temporally upsample a video by a specific factor (x2, x4, ...) -
```
python sample.py --task='video_interp' --image_name=hula_hoop --interpolation_rate=4 --run_name=hula_hoop_interpolate_0
```

By default (if ```output_dir``` isn't specified, the generated images/videos will be saved in the ```./outputs``` folder.

## Citation
If you find our project useful for your work please cite:

```
@article{nikankin2022sinfusion,
  title={SinFusion: Training Diffusion Models on a Single Image or Video},
  author={Nikankin, Yaniv and Haim, Niv and Irani, Michal},
  journal={arXiv preprint arXiv:2211.11743},
  year={2022}
}
```

