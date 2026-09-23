# CPU 快速推理配置：基于 TSN Kinetics-400，将 TenCrop×25 改为 CenterCrop×3
_base_ = ['./tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py']

test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(-1, 256), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
