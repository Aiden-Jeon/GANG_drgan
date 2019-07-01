import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from utils import *
from models import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, help="save directory")
    parser.add_argument('--load_dir', type=str, help="load file directory")
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_num_epochs', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--train_continue', action='store_true')
    args = parser.parse_args()

    train_dataset = DistortionDataset('dis/', 'raw/', True)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    valid_dataset = DistortionDataset('dis/', 'raw/', False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=3, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = args.save_dir

    G = Generator(64, dtype)
    D = Discriminator(64, dtype)
    feature_extractor = FeatureExtractor(models.vgg19(pretrained=True), dtype)  

    bce_loss = nn.BCELoss().type(dtype)
    mse_loss = nn.MSELoss().type(dtype)
    l1_loss = nn.L1Loss().type(dtype)
    G_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.9))

    for valid_realX, valid_fakeY in valid_dataloader:
        valid_realX = Variable(valid_realX).type(dtype)
        valid_fakeY = Variable(valid_fakeY).type(dtype)
        break
    with open(checkpoint_path + 'hist.txt', 'w') as f:
        f.write('') 
    if args.pretrain :
        for epoch in range(args.pretrain_num_epochs):
            print('Starting epoch %d/%d' %(epoch+1, num_epoch))
            total_G_loss = 0
            G.train()
            for realX, targetY in tqdm(train_dataloader):
                batch_size = realX.size()[0]
                
                realX = Variable(realX).type(dtype)
                targetY = Variable(targetY).type(dtype)
                fakeX = G(realX)
                G.zero_grad()

                G_content_loss = mse_loss(fakeX, targetY)
                total_G_loss += G_content_loss.cpu().data

                G_content_loss.backward()
                G_optimizer.step()

    num_epochs = args.num_epochs
    save_every = args.save_every
    start_epoch = 0

    if args.train_continue:
        G_load = torch.load(checkpoint_path + 'G_params.pkl')
        D_load = torch.load(checkpoint_path + 'D_params.pkl')
        G.load_state_dict(G_load['params'])
        D.load_state_dict(D_load['params'])
        start_epoch = G_load['epoch']


    for epoch in range(start_epoch+1, num_epochs+1):
        print('Starting epoch %d/%d' %(epoch, num_epochs))
        tic = time.time()
        G.train()
        D.train()
        total_G_content_loss = 0
        total_G_adversarial_loss = 0
        total_G_loss = 0
        total_D_loss = 0
        for realX, targetY in tqdm(train_dataloader):
            batch_size = realX.size()[0]
            
            realX = Variable(realX).type(dtype)
            targetY = Variable(targetY).type(dtype)
            
            ### Train Discriminator
            # label smoothing
            target_real = Variable(torch.rand(batch_size,1)*0.5 + 0.7).type(dtype)
            target_fake = Variable(torch.rand(batch_size,1)*0.3).type(dtype)
            
            # by WGAN train D 10 times
            fakeX = Variable(G(realX))
            for _ in range(10):
                D.zero_grad()
                target_result = D(targetY).squeeze().view(-1,1)
                fake_result = D(fakeX).squeeze().view(-1,1)
                
                D_target_loss = bce_loss(
                    target_result, 
                    target_real
                )
                D_fake_loss = bce_loss(
                    fake_result, 
                    target_fake
                )
                D_loss = D_target_loss + D_fake_loss
                total_D_loss += D_loss.cpu().data
                D_loss.backward()
                D_optimizer.step()

            ### Train G
            G.zero_grad()

            target_features = feature_extractor(targetY)
            fake_features = feature_extractor(fakeX)
            
            target_low_features = target_features.relu_1_2
            fake_low_features = fake_features.relu_1_2
            target_high_features = target_features.relu_5_2
            fake_high_features = fake_features.relu_5_2
            
            G_content_loss = 0.2*mse_loss(fake_low_features, target_low_features) + 0.8*mse_loss(fake_high_features, target_high_features)
            G_adversarial_loss = bce_loss(D(fakeX).squeeze().view(-1,1), Variable(torch.ones(batch_size, 1)).type(dtype))
            G_loss = 20*G_content_loss + G_adversarial_loss
            
            total_G_content_loss += G_content_loss.cpu().data
            total_G_adversarial_loss += G_adversarial_loss.cpu().data
            total_G_loss += G_loss.cpu().data
            
            G_loss.backward()
            G_optimizer.step()
            
        toc = time.time()
        hist = {
            'epoch' : epoch,
            'total_G_content_loss': round(total_G_content_loss.data.tolist()/len(train_dataloader), 5),
            'total_G_adversarial_loss': round(total_G_adversarial_loss.data.tolist()/len(train_dataloader), 5),
            'total_G_loss' : round(total_G_loss.data.tolist()/len(train_dataloader), 5),
            'total_D_loss': round(total_D_loss.data.tolist()/len(train_dataloader)/10, 5),
            'epoch_time': round(toc-tic, 4),
        }
        save_img(valid_realX, valid_fakeY, checkpoint_path, epoch)
        f = open(checkpoint_path+'/hist.txt', 'a')
        json.dump(hist, f)
        f.write('\n')
        f.close()
        if epoch % save_every == 0:
            torch.save({
                'epoch' : epoch,
                'params' : D.state_dict()
            },'{0}/D_params.pkl'.format(checkpoint_path))
            torch.save({
                'epoch' : epoch,
                'params' : G.state_dict()
            },'{0}/G_params.pkl'.format(checkpoint_path))

if __name__ == '__main__':
    main()