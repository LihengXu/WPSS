from .base_discriminator import BaseDiscriminator


def build_discriminator(cfg):
    discriminator = eval(cfg.DISCRIMINATOR.TYPE)(cfg)
    return discriminator
