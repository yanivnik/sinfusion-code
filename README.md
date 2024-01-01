

# [ICML 2023] SinFusion - Training Diffusion Models on a Single Image or Video
[![project](https://img.shields.io/badge/project-sinfusion-blue)](https://yanivnik.github.io/sinfusion) [![arxiv](https://img.shields.io/badge/arxiv-2211.11743-brightgreen)](https://arxiv.org/abs/2211.11743) [![Conference](https://img.shields.io/badge/ICML-2023-blueviolet)]()

Official PyTorch implementation of *"SinFusion - Training Diffusion Models on a Single Image or Video"*. 


### Data and Checkpoints

You can find training images, training video frames, and some sample checkpoints in the [Dropbox Folder](https://www.dropbox.com/sh/75z6elt62i75kq4/AACFHqvo1L69zyY-JmhWO8eRa?dl=0).

You can download the training images and video frames from the ```images``` folder into a local ```./images``` folder.

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

To download the checkpoints, download the ```checkpoints/lightning_logs``` directory from the Dropbox to a local ```./lightning_logs``` directory. The structure should be, for example - 
```
some/path/sinfusion/lightning_logs/
    balloons.png/
        image_balloons_sample_weights/checkpoints/last.ckpt
    tornado/
        video_tornado_sample_weights_predictor/checkpoints/last.ckpt
        video_tornado_sample_weights_projector/checkpoints/last.ckpt
```
Following this you can directly sample the trained weights (using the default configuration in ```config.py```) with the generation command lines below (notice you need to use the relevant ```run_name```, for example - ```image_balloons_sample_weights``` or ```video_tornado_sample_weights```).

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


### Evaluation

For evaluating using NNFDIV metric (in Section 7.2), please check out the [NNF Diversity Repository](https://github.com/nivha/nnf_diversity).

## Citation
If you find our project useful for your work please cite:

```
@inproceedings{nikankin2022sinfusion,
  title={SinFusion: Training Diffusion Models on a Single Image or Video},
  author={Nikankin, Yaniv and Haim, Niv and Irani, Michal},
  booktitle={International Conference on Machine Learning},
  organization={PMLR},
  year={2023}
}
```

