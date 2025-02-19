from torch.nn import init


def get_loss(c, name):
   
    if name == 'L1':
        return nn.SmoothL1Loss()
    elif name == 'diff':
        sloss1 = nn.MSELoss( reduction='mean')
        sloss2 = nn.MSELoss( reduction='mean')
        def diff_loss(noise1, noise2, audio,p_audio,weight = 0.5):
            loss1 = sloss1(noise1, noise2)
            loss2 = sloss2(audio, p_audio)
          #  print(weight)
          #  print(weight,loss1, loss2)
            loss = weight*loss1+ (1-weight)*loss2
            return loss
        return diff_loss

    return nn.MSELoss( reduction='mean')


def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def get_sqrt_alphas_cumprod_prev():
    betas = make_beta_schedule(schedule='quad', n_timesteps=50,
                                        start=0.5, end=50)
    betas = betas.detach().cpu().numpy() if isinstance(
        betas, torch.Tensor) else betas

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
    sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))
    
    return sqrt_alphas_cumprod_prev

def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)










def train(train_dataloader, params):


    model = get_model(c.model_name, params)
    init_weights(model, init_type='xavier')

    #opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    opt = optim.RAdam(model.parameters(), lr=c.lr)
    learner = DiffWaveLearner(model_dir, model, train_dataloader, opt, params, fp16=True)
    learner.train(max_epochs=c.max_epoch)

    
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1 and classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

    

def get_model(name, params):
    if c.r_con:
        cond_dim = 2
    else:
        cond_dim = 1

 

    from models.conditional_unet1d import ConditionalUnet1D    
    model = ConditionalUnet1D(input_dim = 1, local_cond_dim = cond_dim, diffusion_step_embed_dim = 128,params = AttrDict(params),
                                cond_predict_scale = False,).cuda()

    return model





def normalize_signal(signal):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True, clip= True)#
    
    signal = scaler.fit_transform(signal.T).T
    return signal


def save_signal(data, filename):
    filename = c.RESULT + '/pkl/' + filename
    fu.save(data, filename)


