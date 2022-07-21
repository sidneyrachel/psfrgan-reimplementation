# TODO: Implement BaseModel later
class PSFRGANModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.parsing_model =