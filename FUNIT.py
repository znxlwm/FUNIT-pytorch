import time, itertools
from torch.autograd import grad
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from networks import *
from utils import *
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

class FUNIT(object) :
    def __init__(self, args):
        self.phase = args.phase
        self.name = args.name
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lrG = args.lrG
        self.lrD = args.lrD
        self.weight_decay = args.weight_decay
        self.r1_gamma = args.r1_gamma
        self.beta = args.beta
        self.gan_weight = args.gan_weight
        self.feature_matching_weight = args.feature_matching_weight
        self.reconstruction_weight = args.reconstruction_weight

        """ Generator """
        self.ngf = args.ngf
        self.nmf = args.nmf
        self.ng_downsampling = args.ng_downsampling
        self.nc_downsampling = args.nc_downsampling
        self.ng_upsampling = args.ng_upsampling
        self.ng_res = args.ng_res
        self.n_mlp = args.n_mlp

        """ Discriminator """
        self.ndf = args.ndf
        self.nd_res = args.nd_res

        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.code_dim = args.code_dim
        self.n_class = args.n_class
        self.K = args.K

        self.result_dir = args.result_dir
        self.device = args.device
        self.num_workers = args.num_workers
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Random seed """
        torch.manual_seed(131)
        torch.cuda.manual_seed_all(131)
        np.random.seed(131)

        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 12, self.img_size+12)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size + 12, self.img_size + 12)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.train_folder = ImageFolder(os.path.join(self.dataset, 'train'), train_transform)
        self.train_loader = DataLoader(self.train_folder, batch_size=self.batch_size * 2, shuffle=True, drop_last=True, num_workers=self.num_workers)
        self.test_folder = CustomDataset(os.path.join(self.dataset, 'test'), test_transform, target_num=self.K)
        self.test_loader = DataLoader(self.test_folder, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)

        """ Define Generator, Discriminator """
        self.ConEn = ContentEncoder(input_nc=3, nf=self.ngf, n_downsampling=self.ng_downsampling, n_blocks=self.ng_res).to(self.device)
        self.ClsEn = ClassEncoder(input_nc=3, nf=self.ngf, class_dim=self.code_dim, n_downsampling=self.nc_downsampling).to(self.device)
        self.Dec = Decoder(output_nc=3, nf=self.ngf * 8, nmf=self.nmf, class_dim=self.code_dim, n_upsampling=self.ng_upsampling, n_blocks=self.ng_res, mlp_blocks=self.n_mlp).to(self.device)
        self.Dis = Discriminator(input_nc=3, output_nc=self.n_class, nf=self.ndf, n_blocks=self.nd_res).to(self.device)

        """ init """
        weight_init(self.ConEn)
        weight_init(self.ClsEn)
        weight_init(self.Dec)
        weight_init(self.Dis)
        self.ConEn_, self.ClsEn_, self.Dec_ = deepcopy(self.ConEn), deepcopy(self.ClsEn), deepcopy(self.Dec)
        self.ConEn_.eval(), self.ClsEn_.eval(), self.Dec_.eval()

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)

        """ Optimizer """
        self.G_optim = torch.optim.RMSprop(itertools.chain(self.ConEn.parameters(), self.ClsEn.parameters(), self.Dec.parameters()), lr=self.lrG, weight_decay=self.weight_decay)
        self.D_optim = torch.optim.RMSprop(self.Dis.parameters(), lr=self.lrD, weight_decay=self.weight_decay)

    def train(self):
        """ Writer """
        self.writer = SummaryWriter(log_dir=os.path.join(self.result_dir, self.name, 'log'))

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        self.ConEn.train(), self.ClsEn.train(), self.Dec.train(), self.Dis.train()
        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.name, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.name, 'model'), start_iter)
                print(" [*] Load SUCCESS")

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            try:
                real, label = train_iter.next()
            except:
                train_iter = iter(self.train_loader)
                real, label = train_iter.next()

            onehot_A = torch.zeros(self.batch_size, self.n_class).scatter_(1, label[self.batch_size:].view(-1, 1), 1)
            onehot_B = torch.zeros(self.batch_size, self.n_class).scatter_(1, label[:self.batch_size].view(-1, 1), 1)
            real_A, real_B, label_A, label_B = real[self.batch_size:].to(self.device), real[:self.batch_size].to(self.device), label[self.batch_size:].to(self.device), label[:self.batch_size].to(self.device)
            onehot_A, onehot_B = onehot_A.to(self.device), onehot_B.to(self.device)

            # Update D
            if self.r1_gamma != 0.0:
                real_B.requires_grad_(True)

            self.D_optim.zero_grad()

            real_logit = self.Dis(real_B)
            real_logit = torch.sum(real_logit * onehot_B.view(self.batch_size, self.n_class, 1, 1), 1, keepdim=True)

            D_real_loss = self.gan_weight * torch.nn.ReLU()(1.0 - real_logit).mean()
            D_real_loss.backward(retain_graph=True)

            if self.r1_gamma != 0.0:
                gradients = grad(outputs=real_logit.mean(), inputs=real_B, create_graph=True, retain_graph=True, only_inputs=True)[0]
                r1_penalty = self.r1_gamma * torch.sum(gradients.pow(2), dim=(1, 2, 3)).mean()
                r1_penalty.backward()

            with torch.no_grad():
                con_A = self.ConEn(real_A)
                cls_B = self.ClsEn(real_B)
                fake_A2B = self.Dec(con_A, cls_B)

            fake_logit = self.Dis(fake_A2B.detach())
            fake_logit = torch.sum(fake_logit * onehot_B.view(self.batch_size, self.n_class, 1, 1), 1, keepdim=True)

            D_fake_loss = self.gan_weight * torch.nn.ReLU()(1.0 + fake_logit).mean()
            D_fake_loss.backward()
            
            D_loss = D_real_loss + D_fake_loss + r1_penalty
            self.D_optim.step()

            self.writer.add_scalar('Train/D_loss', D_loss, step)

            # Update G
            con_A, cls_A = self.ConEn(real_A), self.ClsEn(real_A)
            cls_B = self.ClsEn(real_B)

            fake_A2B = self.Dec(con_A, cls_B)

            fake_logit, fake_features = self.Dis.forward_with_features(fake_A2B)
            fake_logit = torch.sum(fake_logit * onehot_B.view(self.batch_size, self.n_class, 1, 1), 1, keepdim=True)

            recon_A = self.Dec(con_A, cls_A)
            recon_logit, recon_features = self.Dis.forward_with_features(recon_A)
            recon_logit = torch.sum(recon_logit * onehot_A.view(self.batch_size, self.n_class, 1, 1), 1, keepdim=True)

            G_fake_loss = -(torch.mean(fake_logit) + torch.mean(recon_logit)) / 2

            G_recon_loss = self.L1_loss(recon_A, real_A)

            _, con_features = self.Dis.forward_with_features(real_A)
            _, cls_features = self.Dis.forward_with_features(real_B)
            G_feature_matching_loss = self.L1_loss(torch.mean(recon_features, dim=[2, 3]), torch.mean(con_features, dim=[2, 3])) + self.L1_loss(torch.mean(fake_features, dim=[2, 3]), torch.mean(cls_features, dim=[2, 3]))

            self.G_optim.zero_grad()
            G_loss = self.gan_weight * G_fake_loss + self.reconstruction_weight * G_recon_loss + self.feature_matching_weight * G_feature_matching_loss
            G_loss.backward()
            self.G_optim.step()

            for pre_params, new_params in zip(self.ConEn_.parameters(), self.ConEn.parameters()):
                pre_params.data.mul_(self.beta).add_(1 - self.beta, new_params.data)

            for pre_params, new_params in zip(self.ClsEn_.parameters(), self.ClsEn.parameters()):
                pre_params.data.mul_(self.beta).add_(1 - self.beta, new_params.data)

            for pre_params, new_params in zip(self.Dec_.parameters(), self.Dec.parameters()):
                pre_params.data.mul_(self.beta).add_(1 - self.beta, new_params.data)

            self.writer.add_scalar('Train/G_loss', G_loss, step)
            self.writer.add_scalar('Train/G_gan_loss', self.gan_weight * G_fake_loss, step)
            self.writer.add_scalar('Train/G_recon_loss', self.reconstruction_weight * G_recon_loss, step)
            self.writer.add_scalar('Train/G_feature_matching_loss', self.feature_matching_weight * G_feature_matching_loss, step)

            print("[%5d/%5d] time: %4.4f D_loss: %.8f, G_loss: %.8f" % (step, self.iteration, time.time() - start_time, D_loss, G_loss))

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.name, 'model'), step)

            if step % self.print_freq == 0:
                params = {}
                params['ConEn'] = self.ConEn.state_dict()
                params['ClsEn'] = self.ClsEn.state_dict()
                params['Dec'] = self.Dec.state_dict()
                params['ConEn_'] = self.ConEn_.state_dict()
                params['ClsEn_'] = self.ClsEn_.state_dict()
                params['Dec_'] = self.Dec_.state_dict()
                params['Dis'] = self.Dis.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.name + '_params_latest.pt'))

            if step % self.print_freq == 0:
                with torch.no_grad():
                    n_sample = 3
                    assert (n_sample <= self.batch_size)
                    output = np.zeros((self.img_size * (self.K + 2), 0, 3))
                    try:
                        target, _ = test_iter.next()
                    except:
                        test_iter = iter(self.test_loader)
                        target, _ = test_iter.next()

                    real = target.to(self.device)
                    real_A = real_A[0].expand((self.batch_size, 3, self.img_size, self.img_size))

                    con_A = self.ConEn_(real_A)
                    cls_code = torch.zeros(0, self.batch_size, self.code_dim).to(self.device)
                    for i in range(real.shape[1] // 3):
                        cls = self.ClsEn_(real[:, 3 * i:3 * (i + 1)])
                        cls_code = torch.cat((cls_code, cls.unsqueeze(0)), 0)

                    cls_code = torch.mean(cls_code, 0)
                    result = self.Dec_(con_A, cls_code)

                    for n in range(n_sample):
                        temp = np.zeros((0, self.img_size, 3))
                        for i in range(real.shape[1] // 3):
                            real_B = real[n, 3 * i:3 * (i + 1)]
                            temp = np.concatenate((temp, RGB2BGR(tensor2numpy(denorm(real_B)))), 0)
                        temp = np.concatenate((temp, RGB2BGR(tensor2numpy(denorm(real_A[n]))), RGB2BGR(tensor2numpy(denorm(result[n])))), 0)
                        output = np.concatenate((output, temp), 1)

                    cv2.imwrite(os.path.join(self.result_dir, self.name, 'img', self.name + '_test_result_%07d.png' % (step)), output * 255.0)

    def save(self, dir, step):
        params = {}
        params['ConEn'] = self.ConEn.state_dict()
        params['ClsEn'] = self.ClsEn.state_dict()
        params['Dec'] = self.Dec.state_dict()
        params['ConEn_'] = self.ConEn_.state_dict()
        params['ClsEn_'] = self.ClsEn_.state_dict()
        params['Dec_'] = self.Dec_.state_dict()
        params['Dis'] = self.Dis.state_dict()
        torch.save(params, os.path.join(dir, self.name + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.name + '_params_%07d.pt' % step))
        self.ConEn.load_state_dict(params['ConEn'])
        self.ClsEn.load_state_dict(params['ClsEn'])
        self.Dec.load_state_dict(params['Dec'])
        self.ConEn_.load_state_dict(params['ConEn_'])
        self.ClsEn_.load_state_dict(params['ClsEn_'])
        self.Dec_.load_state_dict(params['Dec_'])
        self.Dis.load_state_dict(params['Dis'])

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.name, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.name, 'model'), start_iter)
            print(" [*] Load SUCCESS")

        self.ConEn_.eval(), self.ClsEn_.eval(), self.Dec_.eval()
        train_iter, test_iter = iter(self.train_loader), iter(self.test_loader)

        with torch.no_grad():
            n_sample = 3
            assert (n_sample <= self.batch_size)
            output = np.zeros((self.img_size * (self.K + 2), 0, 3))
            try:
                content, _ = train_iter.next()
            except:
                train_iter = iter(self.train_loader)
                content, _ = train_iter.next()

            try:
                target, _ = test_iter.next()
            except:
                test_iter = iter(self.test_loader)
                target, _ = test_iter.next()

            content, target = content[:self.batch_size].to(self.device), target.to(self.device)

            con_A = self.ConEn_(content)
            cls_code = torch.zeros(0, self.batch_size, self.code_dim).to(self.device)
            for i in range(target.shape[1] // 3):
                cls = self.ClsEn_(target[:, 3 * i:3 * (i + 1)])
                cls_code = torch.cat((cls_code, cls.unsqueeze(0)), 0)

            cls_code = torch.mean(cls_code, 0)
            result = self.Dec_(con_A, cls_code)

            for n in range(n_sample):
                temp = np.zeros((0, self.img_size, 3))
                for i in range(target.shape[1] // 3):
                    target_img = target[n, 3 * i:3 * (i + 1)]
                    temp = np.concatenate((temp, RGB2BGR(tensor2numpy(denorm(target_img)))), 0)
                temp = np.concatenate((temp, RGB2BGR(tensor2numpy(denorm(content[n]))), RGB2BGR(tensor2numpy(denorm(result[n])))), 0)
                output = np.concatenate((output, temp), 1)

            cv2.imwrite(os.path.join(self.result_dir, self.name, 'test', self.name + '_test_result.png'), output * 255.0)
