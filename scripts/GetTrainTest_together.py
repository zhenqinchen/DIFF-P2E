import torch.nn.functional as F

def get_rpeak_loss():#c, name
    from monai.losses import FocalLoss
    return FocalLoss(to_onehot_y=True)


class Difflearner:
  def __init__(self, model_dir, model, dataset, optimizer, rpeak_model,*args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer

    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True
    self.noise_schedule = c.noise_schedule
    beta = np.array(self.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = nn.MSELoss(reduction='mean') #get_loss(c, c.loss_name)
    self.summary_writer = None
    
    self.rpeak_model = rpeak_model
    self.rpeak_loss = get_rpeak_loss()
    self.loss_x0 = nn.MSELoss(reduction='mean')
    
  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    rpeak_model_state = self.rpeak_model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'rpeak_model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in rpeak_model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.rpeak_model.load_state_dict(state_dict['rpeak_model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_name= f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)

  def restore_from_checkpoint(self, filename='weights'):
    filename = c.model_name + '_model'
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_epochs=None):
    device = next(self.model.parameters()).device
    min_loss = 1111100000
    loss = 0
    rloss, dloss,x0loss = 0,0,0
    epoch_losses = []  # 保存每个 epoch 的损失
    for epoch in range(max_epochs):
        epoch_loss = 0
        batch_count = 0
        for features in tqdm(self.dataset, desc=f'Epoch {epoch + 1}, loss:{loss:.3f}, rloss:{rloss:.3f},dloss:{dloss:.3f},x0loss:{x0loss:.3f}'):#Epoch {self.step // len(self.dataset)}, loss:{loss}
            epoch_loss = 0
            # features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            loss,rloss,dloss,x0loss = self.train_step(features)
            if torch.isnan(loss).any():
                raise RuntimeError(f'Detected NaN loss at step {self.step}.')

            if min_loss > loss.item():
                min_loss = loss.item()
                self.save_to_checkpoint(filename = model_file)
            epoch_loss += loss.item()
            batch_count += 1
            self.step += 1
        # 计算并保存每个 epoch 的平均损失
        average_epoch_loss = epoch_loss / batch_count
        epoch_losses.append(average_epoch_loss)
    
        
        


  def train_step(self, features):
  #  self.optimizer.zero_grad(set_to_none=True)
    for param in self.model.parameters():
      param.grad = None
    for param in self.rpeak_model.parameters():
      param.grad = None
    device = torch.device('cuda')
   # noisy,audio=features#,_,_ noisy = ppg, audio = ecg
    noisy,audio = features['ppg'].to(device), features['ecg'].to(device)
    rlabel = features['rlabel'].to(device)
        
   # print(noisy.shape, audio.shape)
    audio = audio.squeeze(dim=1)
    noisy = noisy.squeeze(dim=1)

    audio = audio.to(torch.float32)
    noisy = noisy.to(torch.float32)
    rlabel = rlabel.to(torch.float32)

    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)
    rloss,dloss,x0loss = 0,0,0
    with self.autocast:
        if c.r_con:
            pred_rlabel = self.rpeak_model(noisy.unsqueeze(dim=1))
            pred_rlabel = F.sigmoid(pred_rlabel)
     #   print(pred_rlabel.shape, rlabel.shape)
        t = torch.randint(0, len(self.noise_schedule), [N], device=audio.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        # noise_scale = self.sqrt_alphas_cumprod_prev[t+1].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(audio)
        if c.normalize_noise:
            noise = normalize_noise(noise)
        
        
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
        if c.r_con:
           # rlabel = features['rlabel'].to(device)
            predicted = self.model(noisy_audio, noisy,t, rlabel = pred_rlabel.squeeze(1))
        else:
            predicted = self.model(noisy_audio, noisy,t)
        predicted_audio = 1/(noise_scale**0.5)*(noisy_audio-(1-noise_scale)**0.5 * predicted.squeeze(1))
        weight= 0.1
        if c.r_con:
            dloss = self.loss_fn(noise, predicted.squeeze(1))
            rloss = self.rpeak_loss(pred_rlabel,rlabel.unsqueeze(1))
            loss = dloss+ weight*rloss
            if c.loss_name == 'diff':
                x0loss = self.loss_x0(audio, predicted_audio)
                loss = dloss+ weight*rloss + weight*x0loss
        else:
            
            if c.loss_name == 'diff':
                dloss = self.loss_fn(noise, predicted.squeeze(1))
                x0loss = self.loss_x0(audio, predicted_audio)
                loss = dloss + weight*x0loss
            else:
                loss = self.loss_fn(noise, predicted.squeeze(1))
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), None or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss,rloss,dloss,x0loss


def train(train_dataloader):


    model = get_model(c.model_name)
    init_weights(model, init_type='xavier')

    rpeak_model = get_rpeak_model(c, c.rpeak_model_name)

    if c.r_con:
        opt = optim.RAdam(list(model.parameters()) + list(rpeak_model.parameters()), lr=c.lr) #optim.RAdam([*model.parameters(), *rpeak_model.parameters()], lr=c.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=c.lr)

    learner = Difflearner(model_dir, model, train_dataloader, opt,rpeak_model, fp16=True)
    learner.train(max_epochs=c.max_epoch)

    
def predict(noisy_audio=None, model=None, rlabel = None, device=torch.device('cuda')):

        
    with torch.no_grad():

        training_noise_schedule = np.array(c.noise_schedule)
        inference_noise_schedule = training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule

     
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        audio = torch.randn(noisy_audio.shape[0], noisy_audio.shape[-1], device=device)

        noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

        for n in range(len(T) - 1, -1, -1):
         #   print(n)
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            if c.r_con:
                audio = c1 * (audio - c2 * model(audio,noisy_audio, torch.tensor([T[n]], device=audio.device), rlabel = rlabel).squeeze(1))
            else:
                audio = c1 * (audio - c2 * model(audio,noisy_audio, torch.tensor([T[n]], device=audio.device)).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)

                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                audio += sigma * noise

    return audio,_

def normalize_signal(signal):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True, clip= True)#
    
    signal = scaler.fit_transform(signal.T).T
    return signal



def test(test_dataloader):

    origins, refs,syns, rlabels = None,None,None, None
    device = torch.device('cuda')
    is_show = False
    
    
   
    checkpoint = torch.load(f'{model_dir}/{model_file}.pt')

    model = get_model(c.model_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    rpeak_model = get_rpeak_model(c, c.rpeak_model_name)
    rpeak_model.load_state_dict(checkpoint['rpeak_model'])
    rpeak_model.eval()

  
    for i, features in enumerate(test_dataloader):
        noisy_signal,clean_signal = features['ppg'].to(device), features['ecg'].to(device)
        rlabel = features['rlabel'].to(device)
        clean_signal = clean_signal# *0.2
        noisy_signal = noisy_signal.squeeze(dim=1)
        clean_signal = clean_signal.squeeze(dim=1)
        noisy_signal = noisy_signal.to(torch.float32).cuda()
        clean_signal = clean_signal.to(torch.float32).cuda()
        
        if c.using_pred_r_con:
            pred_rlabel = rpeak_model(noisy_signal.unsqueeze(dim=1))
            pred_rlabel = F.sigmoid(pred_rlabel)

            rlabel = pred_rlabel.squeeze(dim=1)
        audio = None
        for k in range(c.sample_k):
            tmp_audio, sr = predict(noisy_signal, model=model, rlabel = rlabel,fast_sampling = fast_sampling)
            if audio is None:
                audio = tmp_audio
            else:
                audio += tmp_audio
        audio /= c.sample_k
        pred_signal = audio
        

 
        clean_signal = clean_signal.detach().cpu().numpy()
        pred_signal = pred_signal.detach().cpu().numpy()
        noisy_signal = noisy_signal.detach().cpu().numpy()
        rlabel = rlabel.detach().cpu().numpy() 
        
        if origins is None:
            origins, refs,syns, rlabels = noisy_signal, clean_signal, pred_signal, rlabel 
        else:
            origins = np.vstack((origins, noisy_signal))
            refs = np.vstack((refs, clean_signal))
            syns = np.vstack((syns, pred_signal))
            rlabels = np.vstack((rlabels, rlabel))
    

    util.evaluate(origins, refs,syns, rlabels,c, align = True)
