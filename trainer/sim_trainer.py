from trainer.trainer import *

class SimTrainer(Trainer):
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
            
            dropped = image.clone()
            channel = np.random.randint(0, self.nchannels)
            dropped[:, channel] = 0

            self.scheduler(self.optimizer, epoch, i, self.best_pred)
            self.optimizer.zero_grad()

            full_output = self.model(image)
            dropped_output = self.model(dropped)

            loss = self.criterion((full_output, dropped_output), target)
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