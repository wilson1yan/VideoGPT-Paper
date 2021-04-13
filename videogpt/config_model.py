import videogpt.models as models
import copy


# BAIR / ViZDoom / RoboNet
# 16 x 64 x 64 -> 8 x 32 x 32
vae_res64_ds222 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(2, 2, 2)
)

# TGIF / UCF 64 x 64
# 16 x 64 x 64 -> 4 x 32 x 32
vae_res64_ds422 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 2, 2)
)

# UCF 128 x 128
# 16 x 128 x 128 -> 4 x 32 x 32
vae_res128_ds444 = dict(
    model_cls='VQVAE',
    num_hiddens=240, num_residual_hiddens=128, embedding_dim=256,
    n_codebooks=1, codes_per_book=1024,
    commitment_cost=0.25, decay=0.99,
    attn_n_heads=2, use_attn=True,
    num_residual_layers=4,
    downsample=(4, 4, 4)
)

# BAIR / ViZDoom / RoboNet
gpt_small = dict(
    model_cls='ImageGPT',
    out_features=512,
    proj_dim=128,
    n_head=4, n_layer=8,
    ff_mult=4,
    dropout=0.2,
    checkpoint=False,
    attn_type='sparse',
    attn_kwargs=dict(attn_dropout=0.),
)

# TGIF / UCF
gpt_large = dict(
    model_cls='ImageGPT',
    out_features=1024,
    proj_dim=128,
    n_head=8, n_layer=20,
    ff_mult=4,
    dropout=0.2,
    checkpoint=True,
    attn_type='sparse',
    attn_kwargs=dict(attn_dropout=0.),
)


configs_str_to_configs = {
    'vae_res64_ds222': vae_res64_ds222,
    'vae_res64_ds422': vae_res64_ds422,
    'vae_res128_ds444': vae_res128_ds444,

    'gpt_small': gpt_small,
    'gpt_large': gpt_large,

    '': dict(),
}


def config_model(*, configs_str, cond_types, **override_kwargs):
    configs = copy.deepcopy(configs_str_to_configs[configs_str])
    configs.update(override_kwargs)

    model_cls = configs.pop('model_cls')
    model = getattr(models, model_cls)(**configs, cond_types=cond_types)

    configs_to_log = {**configs, 'model_cls': model_cls}
    return model, configs_to_log
