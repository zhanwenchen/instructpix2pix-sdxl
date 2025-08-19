#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Harutatsu Akiyama and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import Namespace
from functools import partial
import logging
import math
import os
import warnings
from pathlib import Path
import accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import compile_regions
from datasets import load_dataset, Dataset, IterableDataset
from datasets.utils.logging import set_verbosity_warning as datasets_set_verbosity_warning, set_verbosity_error as datasets_set_verbosity_error
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import StableDiffusionXLInstructPix2PixPipeline
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.logging import set_verbosity_info as diffusers_set_verbosity_info, set_verbosity_error as diffusers_set_verbosity_error
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
from numpy import asarray as np_asarray
from packaging import version
from PIL.Image import open as Image_open
import torch
from torch import (
    float16 as torch_float16,
    float32 as torch_float32,
    bfloat16 as torch_bfloat16,
    no_grad as torch_no_grad,
    cat as torch_cat,
    stack as torch_stack,
    Generator as torch_Generator,
    contiguous_format as torch_contiguous_format,
)
from torch.nn import Conv2d
from torch.nn.functional import mse_loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, CenterCrop, RandomCrop, RandomHorizontalFlip
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection
from transformers.utils.logging import set_verbosity_warning as transformers_set_verbosity_warning, set_verbosity_error as transformers_set_verbosity_error
from wandb import Table as wandb_Table, Image as wandb_Image
from dataset import ImagePairDataset, collate_fn, PairedImageTransform
from utils import parse_args


logger = get_logger(__name__, log_level='INFO')

DATASET_NAME_MAPPING = {
    'fusing/instructpix2pix-1000-samples': ('file_name', 'edited_image', 'edit_prompt'),
}
WANDB_TABLE_COL_NAMES = ['file_name', 'edited_image', 'edit_prompt']
TORCH_DTYPE_MAPPING = {'fp32': torch_float32, 'fp16': torch_float16, 'bf16': torch_bfloat16}
DIRNAME_SOURCE = 'vbi'
DIRNAME_DESTINATION = 'thea'
COLNAME_SOURCE = 'original_image_column'
COLNAME_DESTINATION = 'edited_image_column'
COLNAME_PROMPT = 'edit_prompt_column'


torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


# def preprocess_images(examples: dict, resolution: int, train_transforms: Compose):
#     original_images = np.concatenate(
#         [convert_to_np(image, resolution) for image in examples[COLNAME_SOURCE]]
#     )
#     edited_images = np.concatenate(
#         [convert_to_np(image, resolution) for image in examples[COLNAME_DESTINATION]]
#     )
#     # We need to ensure that the original and the edited images undergo the same
#     # augmentation transforms.
#     images = np.stack((original_images, edited_images))
#     images = torch.as_tensor(images)
#     images = 2 * (images / 255) - 1
#     return train_transforms(images)


# def preprocess_train(examples, resolution: int, train_transforms: Compose, text_encoders, tokenizers):
#     # Preprocess images.
#     preprocessed_images = preprocess_images(examples, resolution, train_transforms)
#     # Since the original and edited images were concatenated before
#     # applying the transformations, we need to separate them and reshape
#     # them accordingly.
#     original_images, edited_images = preprocessed_images
#     original_images = original_images.view(-1, 3, resolution, resolution)
#     edited_images = edited_images.view(-1, 3, resolution, resolution)

#     # Collate the preprocessed images into the `examples`.
#     examples['original_pixel_values'] = original_images
#     examples['edited_pixel_values'] = edited_images

#     # Preprocess the captions.
#     captions = list(examples[COLNAME_PROMPT])
#     prompt_embeds_all, add_text_embeds_all = compute_embeddings_for_prompts(captions, text_encoders, tokenizers)
#     examples['prompt_embeds'] = prompt_embeds_all
#     examples['add_text_embeds'] = add_text_embeds_all
#     return examples


# def collate_fn(examples):
#     breakpoint()
#     original_pixel_values = torch_stack([example['original_pixel_values'] for example in examples])
#     original_pixel_values = original_pixel_values.to(memory_format=torch_contiguous_format, dtype=torch_float32, non_blocking=True)
#     edited_pixel_values = torch_stack([example['edited_pixel_values'] for example in examples])
#     edited_pixel_values = edited_pixel_values.to(memory_format=torch_contiguous_format, dtype=torch_float32, non_blocking=True)
#     prompt_embeds = torch_cat([example['prompt_embeds'] for example in examples], dim=0)
#     add_text_embeds = torch_cat([example['add_text_embeds'] for example in examples], dim=0)
#     return {
#         'original_pixel_values': original_pixel_values,
#         'edited_pixel_values': edited_pixel_values,
#         'prompt_embeds': prompt_embeds,
#         'add_text_embeds': add_text_embeds,
#     }

def log_validation(pipeline, args, accelerator, generator, global_step, is_final_validation=False):
    logger.info(f'Running validation... \n Generating {args.num_validation_images} images with prompt: {args.validation_prompt}.')

    pipeline = pipeline.to(accelerator.device, non_blocking=True)
    pipeline.set_progress_bar_config(disable=True)

    val_save_dir = os.path.join(args.output_dir, 'validation_images')
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    # original_image = (
    #     lambda image_url_or_path: load_image(image_url_or_path)
    #     if urlparse(image_url_or_path).scheme
    #     else Image.open(image_url_or_path).convert('RGB')
    # )(args.val_image_url_or_path)
    original_image = load_image(args.val_image_url_or_path)

    # autocast_ctx = torch.autocast(accelerator.device.type)

    # with autocast_ctx:
    with accelerator.autocast():
        edited_images = []
        # Run inference
        for val_img_idx in range(args.num_validation_images):
            a_val_img = pipeline(args.validation_prompt, image=original_image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, generator=generator).images[0]
            edited_images.append(a_val_img)
            # Save validation images
            a_val_img.save(os.path.join(val_save_dir, f'step_{global_step}_val_img_{val_img_idx}.png'))

    for tracker in accelerator.trackers:
        if tracker.name == 'wandb':
            wandb_table = wandb_Table(columns=WANDB_TABLE_COL_NAMES)
            for edited_image in edited_images:
                wandb_table.add_data(wandb_Image(original_image), wandb_Image(edited_image), args.validation_prompt)
            logger_name = 'test' if is_final_validation else 'validation'
            tracker.log({logger_name: wandb_table})


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = 'text_encoder'):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = text_encoder_config.architectures[0]

    if model_class == 'CLIPTextModel':
        return CLIPTextModel
    if model_class == 'CLIPTextModelWithProjection':
        return CLIPTextModelWithProjection
    raise ValueError(f'{model_class} is not supported.')


def convert_to_np(fpath_image, resolution):
    # breakpoint()
    # if isinstance(fpath_image, str):
        # image = Image_open(image).draft('RGB', (resolution, resolution))
    with Image_open(fpath_image) as image_pil:
        # image_pil = Image_open(image)
        # image_temp.draft('RGB', (resolution, resolution))
        # image.draft(
        image_pil = image_pil.convert('RGB').resize((resolution, resolution))
    # return pil_to_tensor(
    image_np = np_asarray(image_pil).transpose(2, 0, 1)
    return image_np # (3, 256, 256) uint8

    # from torchvision.io import decode_image

    # image_pt = decode_image(image, mode='RGB')
    # from torchvision.transforms.v2.functional import resize
    # image_pt = resize(image_pt, (resolution, resolution))


def main():
    args = parse_args()

    if args.report_to == 'wandb' and args.hub_token is not None:
        raise ValueError('You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token. Please use `huggingface-cli login` to authenticate with the Hub.')

    if args.non_ema_revision is not None:
        deprecate('non_ema_revision!=None', '0.15.0', message='Downloading non_ema weights from revision branches of the Hub is deprecated. Please make sure to use `--variant=non_ema` instead.')
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=args.report_to, project_config=accelerator_project_config)
    accelerator = Accelerator()

    generator = torch_Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets_set_verbosity_warning()
        transformers_set_verbosity_warning()
        diffusers_set_verbosity_info()
    else:
        datasets_set_verbosity_error()
        transformers_set_verbosity_error()
        diffusers_set_verbosity_error()

    # If passed along, set the training seed now.
    if (seed := args.seed) is not None:
        set_seed(seed)

    # Handle the repository creation
    output_dir = args.output_dir
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    vae_path = args.pretrained_model_name_or_path if (pretrained_vae_model_name_or_path := args.pretrained_vae_model_name_or_path) is None else pretrained_vae_model_name_or_path
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae' if pretrained_vae_model_name_or_path is None else None, revision=args.revision, variant=args.variant)
    # vae = torch.compile(vae)
    # vae = compile_regions(vae, mode='reduce-overhead', fullgraph=True)
    # unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet', revision=args.revision, variant=args.variant).compile_repeated_blocks()
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder='unet', revision=args.revision, variant=args.variant)
    # unet.up_blocks = torch.compile(unet.up_blocks)
    # unet.down_blocks = torch.compile(unet.down_blocks)
    # unet = compile_regions(unet, mode='reduce-overhead', fullgraph=True)
    # unet = compile_repeated_blocks(unet)

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info('Initializing the XL InstructPix2Pix UNet from the pretrained UNet.')
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch_no_grad():
        new_conv_in = Conv2d(in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    # TODO: pipe.enable_xformers_memory_efficient_attention (https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/img2img#diffusers.StableDiffusionImg2ImgPipeline.enable_xformers_memory_efficient_attention)
    if args.enable_xformers_memory_efficient_attention:
        assert is_xformers_available(), 'xformers is not available. Make sure it is installed correctly'
        unet.enable_xformers_memory_efficient_attention()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse('0.16.0'):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, 'unet_ema'))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, 'unet'))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, 'unet_ema'), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device, non_blocking=True)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder='unet')
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer = AdamW(unet.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    resolution = args.resolution
    train_transforms = []

    # Add the cropping transform
    if args.center_crop:
        train_transforms.append(CenterCrop(resolution))
    else:
        train_transforms.append(RandomCrop(resolution))

    # Conditionally add the flipping transform
    if args.random_flip:
        train_transforms.append(RandomHorizontalFlip())

    # Compose the final list of transforms
    train_transforms = Compose(train_transforms) if len(train_transforms) > 1 else train_transforms[0]
    train_transforms = PairedImageTransform(train_transforms)
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    # else:
        # data_files = {}
        # if args.train_data_dir is not None:
        #     data_files['train'] = os.path.join(args.train_data_dir, '**')
        # dataset = load_dataset('imagefolder', data_files=data_files, cache_dir=args.cache_dir) # See more about loading custom images at

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # column_names = dataset['train'].column_names
    # column_names = dataset.column_names

    # 6. Get the column names for input/target.
    # dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    # if args.original_image_column is None:
    #     original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #     original_image_column = args.original_image_column
    #     if original_image_column not in column_names:
    #         raise ValueError(f'--original_image_column value {args.original_image_column} needs to be one of: {', '.join(column_names)}')
    # if (edit_prompt_column := args.edit_prompt_column) is None:
    #     edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    # else:
    #     if edit_prompt_column not in column_names:
    #         raise ValueError(f'--edit_prompt_column value {args.edit_prompt_column} needs to be one of: {', '.join(column_names)}')
    # if args.edited_image_column is None:
    #     edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    # else:
    #     edited_image_column = args.edited_image_column
    #     if edited_image_column not in column_names:
    #         raise ValueError(f'--edited_image_column value {args.edited_image_column} needs to be one of: {', '.join(column_names)}')

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    match accelerator.mixed_precision:
        case 'fp16':
            weight_dtype = torch_float16
            warnings.warn(f'{weight_dtype=} may cause nan during vae encoding', UserWarning)
        case 'bf16':
            weight_dtype = torch_bfloat16
            warnings.warn(f'{weight_dtype=} may cause nan during vae encoding', UserWarning)
        case _:
            weight_dtype = torch_float32

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions, tokenizer):
        return tokenizer(captions, max_length=tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids

    # Preprocessing the datasets.
    # train_transforms = Compose(
    #     [
    #         CenterCrop(resolution) if args.center_crop else RandomCrop(resolution),
    #         RandomHorizontalFlip() if args.random_flip else Lambda(lambda x: x),
    #     ]
    # )

    # Load scheduler, tokenizer and models.
    tokenizer_1 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer', revision=args.revision, use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer_2', revision=args.revision, use_fast=False)
    text_encoder_cls_1 = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_2 = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder='text_encoder_2')

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
    text_encoder_1 = text_encoder_cls_1.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder', revision=args.revision, variant=args.variant)
    text_encoder_2 = text_encoder_cls_2.from_pretrained(args.pretrained_model_name_or_path, subfolder='text_encoder_2', revision=args.revision, variant=args.variant)

    # We ALWAYS pre-compute the additional condition embeddings needed for SDXL
    # UNet as the model is already big and it uses two text encoders.
    text_encoder_1.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
    text_encoder_2.to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
    # text_encoder_1.to(dtype=weight_dtype)
    # text_encoder_2.to(dtype=weight_dtype)
    tokenizers = (tokenizer_1, tokenizer_2)
    text_encoders = (text_encoder_1, text_encoder_2)

    # Freeze vae and text_encoders
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Set UNet to trainable.
    unet.train()

    # Adapted from examples.dreambooth.train_dreambooth_lora_sdxl
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.

    # Get null conditioning
    def compute_null_conditioning():
        # null_conditioning_list = []
        # for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
            # null_conditioning_list.append(a_text_encoder(tokenize_captions([''], tokenizer=a_tokenizer).to(accelerator.device, non_blocking=True), output_hidden_states=True).hidden_states[-2])
            # null_conditioning_list.append(a_text_encoder(tokenize_captions([''], tokenizer=a_tokenizer), output_hidden_states=True).hidden_states[-2])
        null_conditioning_list = [a_text_encoder(tokenize_captions([''], tokenizer=a_tokenizer).to(accelerator.device, non_blocking=True), output_hidden_states=True).hidden_states[-2] for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders)]
        # null_conditioning_list = [a_text_encoder(tokenize_captions([''], tokenizer=a_tokenizer), output_hidden_states=True).hidden_states[-2] for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders)]
        return torch_cat(null_conditioning_list, dim=-1)

    def compute_time_ids():
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        original_size = target_size = (resolution, resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.as_tensor([add_time_ids], dtype=weight_dtype, device=accelerator.device)
        # add_time_ids = torch.as_tensor([add_time_ids], dtype=weight_dtype)
        return add_time_ids.repeat(args.train_batch_size, 1)

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=seed).select(range(args.max_train_samples))
    # dataset_train = ImagePairDataset(args.train_data_dir, 'train', partial(preprocess_train, resolution=resolution, train_transforms=train_transforms, text_encoders=text_encoders, tokenizers=tokenizers))
    dataset_train = ImagePairDataset(args.train_data_dir, 'train', resolution, train_transforms)

    # collate_fn=lambda batch: collate_fn(batch, text_encoders, tokenizers)

    collate_fn_new = partial(collate_fn, text_encoders=text_encoders, tokenizers=tokenizers)
    train_dataloader = DataLoader(dataset_train, shuffle=not isinstance(dataset_train, IterableDataset), collate_fn=collate_fn_new, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, pin_memory=False, persistent_workers=args.dataloader_num_workers > 0)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    if args.use_ema:
        ema_unet.to(accelerator.device, non_blocking=True)

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
    else:
        vae.to(accelerator.device, dtype=TORCH_DTYPE_MAPPING[args.vae_precision], non_blocking=True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # breakpoint()
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers('instruct-pix2pix-xl', config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(dataset_train)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if (resume_from_checkpoint := args.resume_from_checkpoint):
        if resume_from_checkpoint != 'latest':
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = sorted((d for d in dirs if d.startswith('checkpoint')), key=lambda x: int(x.split('-')[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f'Checkpoint {resume_from_checkpoint} does not exist. Starting a new training run.')
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f'Resuming from checkpoint {path}')
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split('-')[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc='Steps',
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # null_conditioning, add_time_ids = None, None
    null_conditioning = compute_null_conditioning().to(accelerator.device, dtype=weight_dtype, non_blocking=True)
    add_time_ids = compute_time_ids()
    for _epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for _step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                # if pretrained_vae_model_name_or_path is not None:
                #     edited_pixel_values = batch['edited_pixel_values'].to(dtype=weight_dtype, non_blocking=True)
                # else:
                #     edited_pixel_values = batch['edited_pixel_values']
                # torch.compiler.cudagraph_mark_step_begin()
                latents = vae.encode(batch['edited_pixel_values'].to(device=accelerator.device, dtype=vae.dtype, non_blocking=True)).latent_dist.sample() * vae.config.scaling_factor
                # latents = vae.encode(batch['edited_pixel_values'].to(vae, non_blocking=True)).latent_dist.sample() * vae.config.scaling_factor
                if pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype, non_blocking=True)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.int64)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL additional inputs
                encoder_hidden_states = batch['prompt_embeds']
                add_text_embeds = batch['add_text_embeds']

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                if pretrained_vae_model_name_or_path is not None:
                    original_pixel_values = batch['original_pixel_values'].to(dtype=weight_dtype, non_blocking=True)
                else:
                    original_pixel_values = batch['original_pixel_values']
                original_image_embeds = vae.encode(original_pixel_values).latent_dist.sample()
                if pretrained_vae_model_name_or_path is None:
                    original_image_embeds = original_image_embeds.to(weight_dtype, non_blocking=True)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://huggingface.co/papers/2211.09800.
                if (conditioning_dropout_prob := args.conditioning_dropout_prob) is not None:
                    # random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    random_p = torch.rand((bsz, 1, 1), device=original_image_embeds.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * conditioning_dropout_prob
                    # prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    # image_mask_dtype = original_image_embeds.dtype
                    # image_mask = ~(conditioning_dropout_prob <= random_p < 3 * conditioning_dropout_prob)
                    image_mask = ((random_p < conditioning_dropout_prob) | (random_p >= 3 * conditioning_dropout_prob)).unsqueeze(-1)

                    # image_mask = 1 - (random_p >= conditioning_dropout_prob).to(image_mask_dtype, non_blocking=True) * (random_p < 3 * conditioning_dropout_prob).to(image_mask_dtype, non_blocking=True)
                    # image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch_cat((noisy_latents, original_image_embeds), dim=1)

                # Get the target for loss depending on the prediction type
                if (prediction_type := noise_scheduler.config.prediction_type) == 'epsilon':
                    target = noise
                elif prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f'Unknown prediction type {prediction_type}')

                # Predict the noise residual and compute loss
                added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids': add_time_ids}

                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
                loss = mse_loss(model_pred.to(torch_float32, non_blocking=True), target.to(torch_float32, non_blocking=True), reduction='mean')

                # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_loss = accelerator.reduce(loss.detach(), reduction='mean')
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({'train_loss': train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        # if args.checkpoints_total_limit is not None:
                        #     checkpoints = os.listdir(args.output_dir)
                        #     checkpoints = sorted((d for d in checkpoints if d.startswith('checkpoint')), key=lambda x: int(x.split('-')[1]))

                        #     # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        #     if len(checkpoints) >= args.checkpoints_total_limit:
                        #         num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        #         removing_checkpoints = checkpoints[0:num_to_remove]

                        #         logger.info(f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints')
                        #         logger.info(f'removing checkpoints: {', '.join(removing_checkpoints)}')

                        #         for removing_checkpoint in removing_checkpoints:
                        #             removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        #             shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                        accelerator.save_state(save_path)
                        logger.info(f'Saved state to {save_path}')
            accelerator.wait_for_everyone()
            logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            ### BEGIN: Perform validation every `validation_epochs` steps
            if global_step % args.validation_steps == 0:
                assert args.val_image_url_or_path is not None and args.validation_prompt is not None
                # create pipeline
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    text_encoder=text_encoder_1,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer_1,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )

                log_validation(pipeline, args, accelerator, generator, global_step, is_final_validation=False)

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()
            ### END: Perform validation every `validation_epochs` steps

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            vae=vae,
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
        )

        pipeline.save_pretrained(args.output_dir)

        if (args.val_image_url_or_path is not None) and (args.validation_prompt is not None):
            log_validation(
                pipeline,
                args,
                accelerator,
                generator,
                global_step,
                is_final_validation=True,
            )

    accelerator.end_training()


# def load_image_pair_dataset_split(args: Namespace, split: str) -> IterableDataset:
#     match split:
#         case 'train':
#             dirpath_split = args.train_data_dir
#         case 'valid':
#             dirpath_split = args.validation_data_dir
#         case 'test':
#             dirpath_split = args.test_data_dir
#         case _:
#             raise ValueError(f'Unknown split {split}. Expected one of: train, validation, test.')
#     dirpath_split = Path(dirpath_split)
#     dirpath_input_images = dirpath_split / DIRNAME_SOURCE / split
#     dirpath_edited_images = dirpath_split / DIRNAME_DESTINATION / split

#     fpaths_source = []
#     fpaths_destination = []
#     for dirpath_fnsku_edited_image in dirpath_edited_images.iterdir():
#         fnsku = dirpath_fnsku_edited_image.stem
#         dirpath_fnsku_original_image = dirpath_input_images / fnsku
#         fnsku_vbis = list(dirpath_fnsku_original_image.glob('*.jpg'))
#         assert len(fnsku_vbis) == 1, f'Expected one original image for {fnsku}, but found {len(fnsku_vbis)=} from {dirpath_fnsku_original_image=}.'
#         fpaths_source.append(str(fnsku_vbis[0]))
#         fnsku_theas = list(dirpath_fnsku_edited_image.glob('*.jpg'))
#         assert len(fnsku_theas) == 1, f'Expected one edited image for {fnsku}, but found {len(fnsku_theas)} from {dirpath_fnsku_edited_image=}.'
#         fpaths_destination.append(str(fnsku_theas[0]))

#     dataset = Dataset.from_dict({
#         'file_name': fpaths_source,
#         'edited_image': fpaths_destination,
#         'edit_prompt': [''] * len(fpaths_destination),
#     }, split=split)

    # return dataset


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
