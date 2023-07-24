import os
import json
from tqdm import tqdm
import wandb
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from graphs.models.tracknetv3 import TrackNetv3
from graphs.losses.adaptivewing import AdaptiveWingLoss
from graphs.losses.wbce import WBCE
from graphs.losses.euclidean import EuclideanLoss
from graphs.losses.my_loss import MyLoss

from agents.base_agent import BaseAgent
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import dataset


class TrackNetv3Agent(BaseAgent):
    def __init__(self, config):
        super(TrackNetv3Agent, self).__init__(config)

        self.model_version = config.model_version
        self.model = globals()[self.config.model](self.config)
        print(f"****************** Using {self.config.model} ******************")

        loss_functions = {
            'mse': nn.MSELoss(),
            'euc': EuclideanLoss(),
            'huber': nn.HuberLoss(),
            'wbce': WBCE(config.pos_factor),
            'l1': nn.L1Loss(),
            'adwing': AdaptiveWingLoss(),
            'my_loss': MyLoss(config.pos_factor)
        }

        self.loss = loss_functions[self.config.loss_function]

        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=config.lr,
        #     momentum=config.momentum, 
        #     weight_decay=config.weight_decay
        # )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        

        if config.type == 'auto':
            full_dataset = dataset.GenericDataset.from_dir(config)
        elif config.type == 'image':
            full_dataset = dataset.ImagesDataset(config)
        elif config.type == 'video':
            full_dataset = dataset.VideosDataset(config)
        else:
            raise Exception("type argument must be one of {'auto', 'image', 'video'}")

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=5,
            mode='max',
            factor=0.5,
            threshold=0.02,
            verbose=True
        )

        train_size = int(config.train_size * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=(not config.no_shuffle))
        self.val_loader = DataLoader(test_dataset, batch_size=config.val_batch_size)

        self.images, self.heatmaps = next(iter(self.train_loader))

        self.is_cuda = torch.cuda.is_available()

        if self.is_cuda and config.device == 'cuda':
            self.device = 'cuda'
            torch.cuda.manual_seed_all(self.config.seed)
            print("Operation will be on *****GPU-CUDA***** ")
        
        else:
            self.device = 'cpu'
            torch.manual_seed(self.config.seed)
            print("Operation will be on *****CPU***** ")


        self.model = self.model.to(self.device)
        self.heatmaps = self.heatmaps.to(self.device)
        self.loss = self.loss.to(self.device)

        self.writer = None
        if config.tensorboard:
            self.writer = SummaryWriter()
            self.writer.add_graph(self.model, self.images.to(self.device))

        if config.wandb:
            wandb.init(
                project=config.project_name,
                config=vars(config)
            )
            wandb.watch(self.model, criterion=self.loss, log='all', log_freq=config.log_period)

        print('Loss using zeros: ', self.loss(torch.zeros_like(self.heatmaps, device=self.device), self.heatmaps), '\n')

        with open(os.path.join(config.save_path, "config.json"), "w") as file:
            json.dump(vars(config), file)

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.single_batch_overfit:
                print('Overfitting on a single batch.')
                self.train_loader = [(self.images, self.heatmaps)]
                self.val_loader = None

            else:
                print("Starting training")
            
            self.training_loop()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def training_loop(self):
        best_val_loss = float('inf')
        for epoch in range(self.config.epochs):
            tqdm.write("Epoch: " + str(epoch))
            running_loss = 0.0

            self.model.train()
            pbar = tqdm(self.train_loader)
            for batch_idx, (X, y) in enumerate(pbar):
                X, y = X.to(self.device), y.to(self.device)

                y_pred = self.model(X)
                loss = self.loss(y_pred, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # running loss calculation
                running_loss += loss.item()
                pbar.set_description(f'Loss: {running_loss / (batch_idx+1):.6f}')

                if batch_idx % self.config.log_period == 0:
                    with torch.inference_mode():
                        images = [
                            torch.unsqueeze(y[0,0,:,:], 0).repeat(3,1,1).cpu(),
                            torch.unsqueeze(y_pred[0,0,:,:], 0).repeat(3,1,1).cpu(),
                        ]
                        if self.config.grayscale:
                            images.append(X[0,:,:,:].cpu())
                            res = X[0,:,:,:] * y[0,0,:,:]
                        else:
                            images.append(X[0,(2,1,0),:,:].cpu())
                            res = X[0, (2,1,0),:,:] * y[0,0,:,:]
                        images.append(res.cpu())
                        grid = torchvision.utils.make_grid(images, nrow=1)#, padding=2)
                        if self.config.wandb:
                            wandb_grid = wandb.Image(grid, caption="Image, predicted output and ball mask")
                            wandb.log({
                                'train':{
                                    'RunningLoss': running_loss / (batch_idx+1),
                                    "ImageResult": wandb_grid,
                                } },
                                step = epoch * len(self.train_loader) + batch_idx
                            )
                        if self.config.tensorboard:
                            self.writer.add_image('ImageResult', grid, epoch*len(self.train_loader) + batch_idx)
                            self.writer.add_scalar('RunningLoss/train', running_loss / (batch_idx+1), epoch * len(self.train_loader) + batch_idx)
                        if self.config.save_images:
                            save_image(grid, f'results/epoch_{epoch}_batch{batch_idx}.png')


            if self.val_loader is not None:
                best = False
                val_loss = self.validation_loop()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best = True

                # save the model
                if epoch % self.config.checkpoint_period == self.config.checkpoint_period - 1:
                    if self.config.save_weights_only:
                        tqdm.write('\n--- Saving weights to: ' + str(self.config.save_path))
                        torch.save(self.model.state_dict(), os.path.join(self.config.save_path, 'last.pth'))
                        if best:
                            torch.save(self.model.state_dict(), os.path.join(self.config.save_path, 'best.pth'))
                    else:
                        tqdm.write('\n--- Saving checkpoint to: ' + str(self.config.save_path))
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': val_loss,
                            }, os.path.join(self.config.save_path, 'last.pt'))
                        if best:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': val_loss,
                                }, os.path.join(self.config.save_path, 'best.pt'))
                
                if self.config.tensorboard:
                    self.writer.add_scalars('Loss', {'train': running_loss / len(self.train_loader), 'val': val_loss}, epoch)
                if self.config.wandb:
                    wandb.log({
                        'train/Loss': running_loss / len(self.train_loader),
                        'val/Loss': val_loss,
                    },
                        step = epoch
                    )
                    wandb.save(str(os.path.join(self.config.save_path, 'best.pth')))
                    wandb.save(str(os.path.join(self.config.save_path, 'last.pth')))
                    wandb.save(str(os.path.join(self.config.save_path, 'best.pt')))
                    wandb.save(str(os.path.join(self.config.save_path, 'last.pt')))

    def validation_loop(self):
        self.model.eval()
        loss_sum = 0
        with torch.inference_mode():
            for X, y in tqdm(self.val_loader):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss_sum += self.loss(y_pred, y)
            tqdm.write('Validation loss: ' + str(loss_sum/len(self.val_loader)))

        return loss_sum/len(self.val_loader)