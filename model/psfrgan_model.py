from model.util import build_fpn_model


# TODO: Implement BaseModel later
class PSFRGANModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.fpn_model = build_fpn_model(
            config=config,
            weight_path=config.fpn_pretrained_weight_file
        )