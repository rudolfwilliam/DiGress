from diffusion_model_discrete import DiscreteDenoisingDiffusion
model = DiscreteDenoisingDiffusion.load_from_checkpoint("./outputs/checkpoint_moses.ckpt")