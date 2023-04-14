from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor
import json
import matplotlib.pyplot as plt
import torch
from flamingo_mini.utils import load_url

"""
config = FlamingoConfig(
        lm="facebook/opt",
        clip_model_type = 'openai/clip-vit-base-patch14',
        dim=768,
        dim_visual = 1024,
        xattn_every=1,
        xattn_dim_head=64,
        xattn_heads=8,
        xattn_ff_mult= 4,
        xattn_act= 'sqrelu',
        resampler_depth= 6,
        resampler_dim_head = 64,
        resampler_heads= 8 ,
        resampler_num_latents = 64,
        resampler_num_time_embeds = 4,
        resampler_ff_mult = 4,
        resampler_act= 'sqrelu',
        freeze_language_model = True,
        freeze_vision_model = True
       )
model = FlamingoModel(config)
model.to("cpu")
processor = FlamingoProcessor(config, device=device)
model.load("./flamingo-coco2/checkpoint-1688/rng_state_2.pth")
model.eval()


image = load_url('https://raw.githubusercontent.com/rmokady/CLIP_prefix_caption/main/Images/CONCEPTUAL_02.jpg')
caption = model.generate_captions(processor, images=[image])
print('generated caption:')
print(caption)
"""

model = FlamingoModel.from_pretrained("flamingo-coco2/checkpoint-1688/") #WORKS
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
processor = FlamingoProcessor(model.config)
#example 1
image = load_url('https://raw.githubusercontent.com/rmokady/CLIP_prefix_caption/main/Images/CONCEPTUAL_02.jpg')
caption = model.generate_captions(processor, images=[image])
print('generated caption:')
print(caption)
#example2
image = load_url("https://i.natgeofe.com/k/b33a3db5-c82c-402f-90a5-0b11494d7739/quokka-OG.jpg")
caption = model.generate_captions(processor, images=[image])
print('generated caption:')
print(caption)
#example3
image = load_url("https://www.aacr.org/wp-content/uploads/2020/03/Thorax_CT_peripheres_Brronchialcarcinom_li_OF_600x450.jpg")
caption = model.generate_captions(processor, images=[image])
print('generated caption:')
print(caption)
#example4
image = load_url("https://ars.els-cdn.com/content/image/1-s2.0-S2211568414000874-gr11.jpg")
caption = model.generate_captions(processor, images=[image])
print('generated caption:')
print(caption)
#example5
image = load_url("https://ctisus.com/resources/library/teaching-files/chest/358000.jpg")
caption = model.generate_captions(processor, images=[image])
print('generated caption:')
print(caption)


