# -*- coding: utf-8 -*-
"""
@Time: 2021/9/11 13:14
@Auth: Rongshan Chen
@File: utils.py
@IDE:PyCharm
@Motto: Happy coding, Thick hair
@Email: 904620522@qq.com
"""
import os
import sys
import numpy as np
import numpy as np
import imageio
import json

import cv2


# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** np.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = np.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        out = list()
        for fn in self.embed_fns:
            temp = fn(inputs)
            out.append(temp)

        return np.concatenate([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    # if i == -1:
    #     return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [np.sin, np.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


def call_Fourier_transform(input, config=None):
    if config == None:
        config = {"multires_views": 4,
                  "i_embed": 0
                  }
    embed, out_dim = get_embedder(config["multires_views"], config["i_embed"])
    out = embed(input)
    return out, out_dim
