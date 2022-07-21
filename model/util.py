def construct_parsing_model(
        config,
        in_size=512,
        out_size=512,
        min_feat_size=32,
        relu_type='LeakyReLU',
        is_train=True,
        weight_path=None
):
