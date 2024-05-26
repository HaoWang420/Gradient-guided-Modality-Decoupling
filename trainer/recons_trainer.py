from trainer.trainer import *

class ReconsTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

    def training(self, epoch):
        self.model.train()

        train_loss = 0.0

        self.train_loader.sampler.set_epoch(epoch)
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()
            
            channel = np.random.randint(0, self.nchannels)
            image[:, channel] = 0

            self.optimizer.zero_grad()

            output, loss = self.forward_batch(image, target)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            tbar.set_description(f'Train loss: {train_loss/(i+1):.3f}')

            if self.cuda_device == 0:
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        if self.cuda_device == 0:
            print('[Epoch: {}]'.format(epoch))
            print('Loss: {:.3f}'.format(train_loss))
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
    
    def resume(self):
        super().resume()
        if not os.path.isfile(self.args.trainer.seg_model.path):
            raise FileNotFoundError(f"=> no checkpoint found at {self.args.trainer.seg_model.path}")
        state_dict = torch.load(self.args.trainer.seg_model.path)
        if self.args.cuda:
            self.seg_model.module.load_state_dict(state_dict['state_dict'])
        else:
            self.seg_model.load_state_dict(state_dict['state_dict'])

    def _init_model(self, nchannel, nclass):
        # Define network
        model = build_model(self.args.model, 
                            nclass=1, 
                            nchannels=nchannel, 
                            model=self.args.model.name)

        seg_model = build_model(self.args.trainer.seg_model, 
                                nclass=nclass,
                                nchannels=nchannel,
                                model=self.args.trainer.seg_model.name
                                )

        train_params = None
        train_params = [{'params': model.parameters(), 'lr': self.args.optim.lr}]

        # Define Optimizer
        if self.args.optim.name == 'sgd':
            optimizer = torch.optim.SGD(train_params)
        elif self.args.optim.name == 'adam':
            optimizer = torch.optim.Adam(train_params, weight_decay=self.args.optim.weight_decay)
        
        self.model, self.optimizer = model, optimizer
        self.seg_model = seg_model

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.cuda_device])
            self.seg_model = self.seg_model.cuda()
            self.seg_model = torch.nn.parallel.DistributedDataParallel(self.seg_model, device_ids=[self.cuda_device])
    
    def forward_batch(self, image, target):
        recons_target = image[:, 1:2].clone()
        recons = self.model(image)
        t1, _, t2, flair = torch.split(image, split_size_or_sections=1, dim=1)

        seg_input = torch.cat([t1, recons, t2, flair], dim=1)
        output = self.seg_model(seg_input)

        loss = self.criterion((recons, output), (recons_target, target))

        return output, loss
    
    def predict(self, image, channel=-1):
        recons = self.model(image)

        # replace t1ce
        if channel == 1:
            image[:, 1, :, :] = recons

        output = self.seg_model(image)

        return output
