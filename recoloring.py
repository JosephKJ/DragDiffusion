from argparse import ArgumentParser
from PIL import Image
from utils.lora_utils import train_lora
from utils.ui_utils import preprocess_image
from diffusers import DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from pytorch_lightning import seed_everything
from drag_pipeline import DragPipeline
from utils.attn_utils import MutualSelfAttentionControl, register_attention_editor_diffusers
from torchvision.utils import save_image

import torch
import torch.nn.functional as F
import numpy as np
import os
import datetime


def main():
    parser = ArgumentParser()
    parser.add_argument('--image-path', default='./images/1.jpeg', type=str)
    parser.add_argument('--prompt', default='', type=str)
    parser.add_argument('--neg-prompt', default='', type=str)
    parser.add_argument('--model-path', default='runwayml/stable-diffusion-v1-5', type=str)
    parser.add_argument('--vae-path', default='default', type=str) # stabilityai/sd-vae-ft-mse
    parser.add_argument('--lora-path', default='./lora_tmp', type=str)
    parser.add_argument('--save-dir', default='./results', type=str)
    parser.add_argument('--lora-step', default=60, type=int)
    parser.add_argument('--lora-lr', default=0.0005, type=int)
    parser.add_argument('--lora-batch_size', default=4, type=int)
    parser.add_argument('--lora-rank', default=16, type=int)
    parser.add_argument('--inversion-strength', default=0.75, type=float)
    parser.add_argument('--lam', default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n-inference-step', default=50, type=int)
    parser.add_argument('--r-m', default=1, type=int)
    parser.add_argument('--r-p', default=3, type=int)
    parser.add_argument('--n-pix-step', default=40, type=int)
    parser.add_argument('--latent-lr', default=0.01, type=float)
    parser.add_argument('--guidance-scale', default=1.0, type=float)
    parser.add_argument('--start-step', default=0, type=int)
    parser.add_argument('--start-layer', default=10, type=int)

    parser.add_argument('--do-lora-training', default=True, type=bool)

    args = parser.parse_args()

    image = Image.open(args.image_path) # .convert("RGB")
    image = np.array(image)

    if args.do_lora_training:
        print('LoRA training is enabled.')
        train_lora(image, args.prompt, args.model_path, args.vae_path, args.lora_path, args.lora_step, args.lora_lr,
                   args.lora_batch_size, args.lora_rank, None)
    else:
        print('LoRA training is disabled.')
        args.lora_path = ""

    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model = DragPipeline.from_pretrained(args.model_path, scheduler=scheduler).to(device)
    model.modify_unet_forward()

    if args.vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(args.vae_path).to(model.vae.device, model.vae.dtype)


    # Book-keeping
    seed_everything(args.seed)
    args.n_actual_inference_step = round(args.inversion_strength * args.n_inference_step)
    args.unet_feature_idx = [3]
    full_h, full_w = image.shape[:2]
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)
    print(args)

    source_image = preprocess_image(image, device)

    # set LoRA
    if args.lora_path == "":
        print("Applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("Applying LoRA: " + args.lora_path)
        model.unet.load_attn_procs(args.lora_path)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(source_image,
                               args.prompt,
                               guidance_scale=args.guidance_scale,
                               num_inference_steps=args.n_inference_step,
                               num_actual_inference_steps=args.n_actual_inference_step)

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(start_step=args.start_step,
                                        start_layer=args.start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    if args.lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    # inference the synthesized image
    gen_image = model(
        prompt=args.prompt,
        neg_prompt=args.neg_prompt,
        batch_size=2, # batch size is 2 because we have reference init_code and updated init_code
        latents=torch.cat([invert_code, invert_code], dim=0),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step
        )[1].unsqueeze(dim=0)

    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

    # save the original image, user editing instructions, synthesized image
    save_result = torch.cat([
        source_image * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        torch.ones((1,3,full_h,25)).cuda(),
        gen_image[0:1]
    ], dim=-1)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_image(save_result, os.path.join(args.save_dir, save_prefix + '.png'))

    print('Done.')


if __name__ == "__main__":
    main()
