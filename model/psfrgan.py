from model.util import build_fpn, build_generator, build_discriminator
from variable.model import GenDisNormTypeEnum


# TODO: Implement BaseModel later
class PSFRGAN(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.fpn_model = build_fpn(
            config=config,
            weight_path=config.fpn_pretrained_weight_file
        )
        self.gen_model = build_generator(
            config=config,
            weight_norm_type=GenDisNormTypeEnum.SPECTRAL
        )

        if self.is_train:
            self.disc_model = build_discriminator(
                config=config,
                in_channel=config.discriminator_in_channel,
                weight_norm_type=GenDisNormTypeEnum.SPECTRAL
            )

            self.vgg_model = loss.PCPFeat(weight_path='./pretrain_models/vgg19-dcbb9e9d.pth').to(opt.device)
            if len(opt.gpu_ids) > 0:
                self.vgg_model = torch.nn.DataParallel(self.vgg_model, opt.gpu_ids, output_device=opt.device)

        self.model_names = ['G']
        self.loss_names = ['Pix', 'PCP', 'G', 'FM', 'D', 'SS'] # Generator loss, fm loss, parsing loss, discriminator loss
        self.visual_names = ['img_LR', 'img_HR', 'img_SR', 'ref_Parse', 'hr_mask']
        self.fm_weights = [1**x for x in range(opt.D_num)]

        if self.isTrain:
            self.model_names = ['G', 'D']
            self.load_model_names = ['G', 'D']

            self.criterionParse = torch.nn.CrossEntropyLoss().to(opt.device)
            self.criterionFM = loss.FMLoss().to(opt.device)
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.device)
            self.criterionPCP = loss.PCPLoss(opt)
            self.criterionPix= nn.L1Loss()
            self.criterionRS = loss.RegionStyleLoss()

            self.optimizer_G = optim.Adam([p for p in self.netG.parameters() if p.requires_grad], lr=opt.g_lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = optim.Adam([p for p in self.netD.parameters() if p.requires_grad], lr=opt.d_lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]