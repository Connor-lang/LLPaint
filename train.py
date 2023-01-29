import torch
from PIL import Image
import torchvision.transforms as transforms
import random
import torch.nn as nn 
import os
import math
import cv2
import numpy as np 
from torch.nn import functional as F 
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
import time
import datetime
from torch.autograd import Variable
import functools
import torchvision
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tensorboard_folder = './tb_logs'
checkpoints_folder = './checkpoints'
image_folder = './checkpoints/images'
sample_folder = './samples'

if not os.path.exists(tensorboard_folder):
  os.makedirs(tensorboard_folder)
if not os.path.exists(checkpoints_folder):
  os.makedirs(checkpoints_folder)
if not os.path.exists(image_folder):
  os.makedirs(image_folder)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def store_dataset(dir):
    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, filenames in sorted(os.walk(dir)):
        for filename in filenames:
            if is_image_file(filename):
                path = os.path.join(root, filename)
                img = Image.open(path).convert('RGB')
                images.append(img)
                all_path.append(path)
                
    return images, all_path

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def initialize(self, opt):
        pass
    
def get_transform():
    transform_list = []
    zoom = 1 + 0.1 * random.randint(0, 4)
    osize = [int(400 * zoom), int(600 * zoom)]
    transform_list.append(transforms.Resize(osize, transforms.functional.InterpolationMode.BICUBIC))
    transform_list.append(transforms.RandomCrop(256))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor(), 
                        transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, root):
        self.root = root
        pass

    def load_data():
        return None

class UnalignedDataset(BaseDataset):
    def initialize(self, root):
        self.root = root

        self.dir_A = os.path.join(self.root, 'trainA')
        self.dir_B = os.path.join(self.root, 'trainB')

        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = get_transform()
    
    def __getitem__(self, index):
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img_np = np.array(A_img)
        B_img_np = np.array(B_img)

        A_img_resize = cv2.resize(A_img_np, (256, 256))
        B_img_resize = cv2.resize(B_img_np, (256, 256))
        
        A_edge = cv2.Canny(A_img_resize, 100, 100)
        A_edge = A_edge.reshape((1, ) + A_edge.shape).astype(np.float32)
        mask = self.luminosity_mask(A_img_resize)

        A_img = torch.from_numpy(A_img_resize.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        B_img = torch.from_numpy(B_img_resize.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()        
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        input_img = A_img

        return {'A': A_img, 'B': B_img, 'A_edge': A_edge, 'mask': mask, 'input_img': input_img, 
                'A_paths': A_path, 'B_paths': B_path}
    
    def lum(self, r, g, b):
      return math.sqrt(.241 * r + .691 * g + .068 * b)

    def luminosity_mask(self, img):
      pixellist = np.vstack(img).tolist()
      sorted_pixellist = sorted(pixellist, key = lambda rgb: self.lum(*rgb))
      lower_bound = np.array(sorted_pixellist[int(len(sorted_pixellist) * 0.1)])
      upper_bound = np.array(sorted_pixellist[int(len(sorted_pixellist) * 0.9)])
      mask = cv2.inRange(img, lower_bound, upper_bound)
      return mask.reshape((1, ) + mask.shape).astype(np.float32)

    def __len__(self):
        return max(self.A_size, self.B_size)

def CreateDataset(root):
    dataset = UnalignedDataset()
    dataset.initialize(root)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def initialize(self, root):
        BaseDataLoader.initialize(self, root)
        self.dataset = CreateDataset(root)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=not False,
            num_workers=int(2))
            
    def load_data(self):
        return self.dataloader
    
    def __len__(self):
        return min(len(self.dataset), math.inf)

def CreateDataLoader(root):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(root)
    return data_loader

def _init_conv_layer(conv, activation, mode='fan_out'):
    if isinstance(activation, nn.LeakyReLU):
        torch.nn.init.kaiming_uniform_(conv.weight, a=activation.negative_slope, nonlinearity='leaky_relu', mode=mode)
    elif isinstance(activation, (nn.ReLU, nn.ELU)):
        torch.nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu', mode=mode)
    else:
        pass
    if conv.bias != None:
        torch.nn.init.zeros_(conv.bias)

class GConv(nn.Module):
  def __init__(self, cnum_in, cnum_out, ksize, stride=1, padding='auto', rate=1, activation=nn.ELU(), bias=True):
    super().__init__()
    padding = rate * (ksize - 1) // 2 if padding == 'auto' else padding 
    self.activation = activation
    self.cnum_out = cnum_out
    num_conv_out = cnum_out if self.cnum_out == 3 or self.activation is None else 2 * cnum_out
    self.conv = nn.Conv2d(cnum_in, num_conv_out, kernel_size = ksize, stride = stride, padding = padding, dilation = rate, bias = bias)
    _init_conv_layer(self.conv, activation = self.activation)
    self.ksize = ksize
    self.stride = stride
    self.rate = rate
    self.padding = padding 

  def forward(self, x):
    x = self.conv(x)
    if self.cnum_out == 3 or self.activation is None:
      return x
    x, y = torch.split(x, self.cnum_out, dim = 1)
    self.activation(x)
    y = torch.sigmoid(y)
    x = x * y 
    return x

class GDownsamplingBlock(nn.Module):
    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_downsample = GConv(cnum_in, cnum_hidden, 3, 2)
        self.conv2 = GConv(cnum_hidden, cnum_out, 3, 1)

    def forward(self, x):
        x = self.conv1_downsample(x)
        x = self.conv2(x)
        return x

class GDeConv(nn.Module):
    def __init__(self, cnum_in,
                 cnum_out,
                 padding=1):
        super().__init__()
        self.conv = GConv(cnum_in, cnum_out, 3, 1,
                          padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=False)
        x = self.conv(x)
        return x

class GUpsamplingBlock(nn.Module):
    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_upsample = GDeConv(cnum_in, cnum_hidden)
        self.conv2 = GConv(cnum_hidden, cnum_out, 3, 1)

    def forward(self, x):
        x = self.conv1_upsample(x)
        x = self.conv2(x)
        return x

class CoarseGenerator(nn.Module):
  def __init__(self, cnum_in, cnum):
    super().__init__()
    self.conv1 = GConv(cnum_in, cnum//2, 5, 1, padding=2)
    self.down_block1 = GDownsamplingBlock(cnum//2, cnum)
    self.down_block2 = GDownsamplingBlock(cnum, 2*cnum)

    self.conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1)
    self.conv_bn2 = GConv(2*cnum, 2*cnum, 3, rate=2, padding=2)
    self.conv_bn3 = GConv(2*cnum, 2*cnum, 3, rate=4, padding=4)
    self.conv_bn4 = GConv(2*cnum, 2*cnum, 3, rate=8, padding=8)
    self.conv_bn5 = GConv(2*cnum, 2*cnum, 3, rate=16, padding=16)
    self.conv_bn6 = GConv(2*cnum, 2*cnum, 3, 1)
    self.conv_bn7 = GConv(2*cnum, 2*cnum, 3, 1)

    self.up_block1 = GUpsamplingBlock(2*cnum, cnum)
    self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)

    self.conv_to_rgb = GConv(cnum//4, 3, 3, 1, activation=None)
    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.conv1(x)

    x = self.down_block1(x)
    x = self.down_block2(x)

    x = self.conv_bn1(x)
    x = self.conv_bn2(x)
    x = self.conv_bn3(x)
    x = self.conv_bn4(x)
    x = self.conv_bn5(x)
    x = self.conv_bn6(x)
    x = self.conv_bn7(x)

    x = self.up_block1(x)
    x = self.up_block2(x)

    x = self.conv_to_rgb(x)
    x = self.tanh(x)
    return x

class FineGenerator(nn.Module):
  def __init__(self, cnum):
    super().__init__()
    self.conv_conv1 = GConv(3, cnum//2, 5, 1, padding=2)

    self.conv_down_block1 = GDownsamplingBlock(
        cnum//2, cnum, cnum_hidden=cnum//2)
    self.conv_down_block2 = GDownsamplingBlock(
        cnum, 2*cnum, cnum_hidden=cnum)

    self.conv_conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1)
    self.conv_conv_bn2 = GConv(2*cnum, 2*cnum, 3, rate=2, padding=2)
    self.conv_conv_bn3 = GConv(2*cnum, 2*cnum, 3, rate=4, padding=4)
    self.conv_conv_bn4 = GConv(2*cnum, 2*cnum, 3, rate=8, padding=8)
    self.conv_conv_bn5 = GConv(2*cnum, 2*cnum, 3, rate=16, padding=16)

    self.ca_conv1 = GConv(3, cnum//2, 5, 1, padding=2)

    self.ca_down_block1 = GDownsamplingBlock(
        cnum//2, cnum, cnum_hidden=cnum//2)
    self.ca_down_block2 = GDownsamplingBlock(cnum, 2*cnum)

    self.ca_conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1, activation=nn.ReLU())
    self.ca_conv_bn4 = GConv(2*cnum, 2*cnum, 3, 1)
    self.ca_conv_bn5 = GConv(2*cnum, 2*cnum, 3, 1)

    self.conv_bn6 = GConv(4*cnum, 2*cnum, 3, 1)
    self.conv_bn7 = GConv(2*cnum, 2*cnum, 3, 1)

    self.up_block1 = GUpsamplingBlock(2*cnum, cnum)
    self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)

    self.conv_to_rgb = GConv(cnum//4, 3, 3, 1, activation=None)
    self.tanh = nn.Tanh()

  def forward(self, x, mask):
    xnow = x

    x = self.conv_conv1(xnow)
    x = self.conv_down_block1(x)
    x = self.conv_down_block2(x)

    x = self.conv_conv_bn1(x)
    x = self.conv_conv_bn2(x)
    x = self.conv_conv_bn3(x)
    x = self.conv_conv_bn4(x)
    x = self.conv_conv_bn5(x)
    x_hallu = x

    x = self.ca_conv1(xnow)
    x = self.ca_down_block1(x)
    x = self.ca_down_block2(x)

    x = self.ca_conv_bn1(x)
    x = self.ca_conv_bn4(x)
    x = self.ca_conv_bn5(x)
    pm = x

    x = torch.cat([x_hallu, pm], dim=1)

    x = self.conv_bn6(x)
    x = self.conv_bn7(x)

    x = self.up_block1(x)
    x = self.up_block2(x)

    x = self.conv_to_rgb(x)
    x = self.tanh(x)

    return x

def output_to_image(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out

class Generator(nn.Module):
  def __init__(self, cnum_in=5, cnum=48):
    super().__init__()
    self.stage1 = CoarseGenerator(cnum_in, cnum)
    self.stage2 = FineGenerator(cnum)
    self.eval()

  def forward(self, x, mask):
    x_in = x
    x_stage1 = self.stage1(x)
    x = x_stage1 * mask + x_in[:, 0:3, :, :] * (1. - mask)
    x_stage2 = self.stage2(x, mask)
    return x_stage1, x_stage2

  @torch.inference_mode()
  def infer(self, image, mask, return_values=['inpainted', 'stage1'], device='cuda'):
    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    image = (image*2 - 1.)
    mask = (mask > 0.).to(dtype=torch.float32)

    image_masked = image * (1.-mask)  
    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x*mask], dim=1)
    x_stage1, x_stage2 = self.forward(x, mask)
    image_compl = image * (1.-mask) + x_stage2 * mask
    output = []
    for return_value in return_values:  
      if return_value.lower() == 'stage1':
        output.append(output_to_image[x_stage1])
      elif return_value.lower() == 'stage2':
                output.append(output_to_image(x_stage2))
      elif return_value.lower() == 'inpainted':
          output.append(output_to_image(image_compl))
      else:
        print(f'Invalid return value: {return_value}')

    return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = 256 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

def pt_to_image(img):
    return img.detach_().cpu().mul_(0.5).add_(0.5)

def save_states(fname, gen, dis, g_optimizer, d_optimizer, n_iter):
    state_dicts = {'G': gen.state_dict(),
                   'D': dis.state_dict(),
                   'G_optim': g_optimizer.state_dict(),
                   'D_optim': d_optimizer.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"./checkpoints/{fname}")
    print("Saved state dicts!")

if __name__ == "__main__":
    print("Creating dataloader")
    data_loader = CreateDataLoader('./dataset')
    dataset = data_loader.load_data()
    adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    losses_log = {'d_loss':   [],
                'g_loss':   [],
                'ae_loss':  [],
                'ae_loss1': [],
                'ae_loss2': [],
                }
    losses = {}

    writer = SummaryWriter("./tb_logs")

    time0 = time.time()
    prev_time = time.time()

    for epoch in range(1000):
        for batch_idx, data in enumerate(dataset):

            A_img = data['A'].to(device)
            B_img = data['B'].to(device)
            A_edge = data['A_edge'].to(device)
            mask = data['mask'].to(device)
            input_img = data['input_img'].to(device)
            image_paths = data['A_paths']

            valid = Variable(Tensor(A_img.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(A_img.shape[0], 1).fill_(0.0), requires_grad=False)

            batch_incomplete = A_img * (1. - mask)
            x = torch.cat([batch_incomplete, A_edge, mask], axis = 1)

            x1, x2 = generator(x, mask)
            batch_predicted = x2

            batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)

            optimizer_G.zero_grad()

            losses['ae_loss1'] = 1 * torch.mean((torch.abs(A_img - x1)))
            losses['ae_loss2'] = 1 * torch.mean((torch.abs(A_img - x2)))
            losses['ae_loss'] = losses['ae_loss1'] + losses['ae_loss2']

            d_gen = discriminator(batch_predicted)
            losses['g_loss'] = adversarial_loss(d_gen, valid)
            losses['g_loss'] += losses['ae_loss']

            losses['g_loss'].backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_pred = discriminator(B_img)
            fake_pred = discriminator(batch_complete.detach())

            real_loss = adversarial_loss(real_pred - fake_pred.mean(0, keepdim = True), valid)
            fake_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim = True), fake)

            d_loss = (real_loss + fake_loss) / 2
            losses['d_loss'] = d_loss

            losses['d_loss'].backward()
            optimizer_D.step()

            for k in losses_log.keys():
                losses_log[k].append(losses[k].item())

            if epoch % 100 == 0:
                dt = time.time() - time0
                batches_done = epoch * len(data_loader) + batch_idx
                batches_left = 40 * len(data_loader) - batches_done
                time_left = datetime.timedelta(seconds = batches_left * (time.time() - time0))
                print(f"@iter {epoch}: {(100 / dt):.4f} it/s")
                print(f"time left: {time_left}")
                time0 = time.time()

                for k, loss_log in losses_log.items():
                    loss_log_mean = sum(loss_log)/len(loss_log)
                    print(f"{k}: {loss_log_mean:.4f}")
                    writer.add_scalar(f"losses/{k}", loss_log_mean, global_step=epoch)
            
            if epoch % 500 == 0:
                viz_images = [pt_to_image(batch_complete), pt_to_image(x1), pt_to_image(x2)]
                img_grids = [torchvision.utils.make_grid(images[:10], nrow=2) for images in viz_images]
                writer.add_image(
                    "Inpainted", img_grids[0], global_step=epoch, dataformats="CHW")
                writer.add_image(
                    "Stage 1", img_grids[1], global_step=epoch, dataformats="CHW")
                writer.add_image(
                    "Stage 2", img_grids[2], global_step=epoch, dataformats="CHW")
                
            if epoch % 500 == 0:
                viz_images = [pt_to_image(A_img), pt_to_image(batch_complete)]
                img_grids = [torchvision.utils.make_grid(images[:10], nrow=2) for images in viz_images]
                torchvision.utils.save_image(img_grids,  f"./checkpoints/images/iter_{epoch}.png", nrow=2)

            if epoch % 100 == 0:
                save_states(f"states.pth", generator, discriminator, optimizer_G, optimizer_D, epoch)

            if epoch % 5000 == 0:
                save_states(f"states_{epoch}.pth", generator, discriminator, optimizer_G, optimizer_D, epoch)