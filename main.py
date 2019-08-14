from FUNIT import FUNIT
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of TSGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--name', type=str, default='FUNIT', help='project_name')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image save frequency')
    parser.add_argument('--save_freq', type=int, default=5000, help='The number of model save frequency')

    parser.add_argument('--lrG', type=float, default=0.0001, help='The learning rate for generator')
    parser.add_argument('--lrD', type=float, default=0.0001, help='The learning rate for discriminator')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--r1_gamma', type=float, default=10, help='The r1_gamma')
    parser.add_argument('--beta', type=float, default=0.999, help='The weight for EMA')
    parser.add_argument('--gan_weight', type=float, default=1.0, help='Weight about GAN')
    parser.add_argument('--feature_matching_weight', type=float, default=1.0, help='Weight about feature matching')
    parser.add_argument('--reconstruction_weight', type=float, default=0.1, help='Weight about reconstruction')

    parser.add_argument('--ngf', type=int, default=64, help='The number of base filter for generator')
    parser.add_argument('--ndf', type=int, default=64, help='The number of base filter for discriminator')
    parser.add_argument('--nmf', type=int, default=256, help='The number of base filter for mlp')
    parser.add_argument('--ng_downsampling', type=int, default=3, help='The number of downsampling layers for content encoder')
    parser.add_argument('--nc_downsampling', type=int, default=4, help='The number of downsampling layers for class encoder')
    parser.add_argument('--ng_upsampling', type=int, default=3, help='The number of upsampling layers for decoder')
    parser.add_argument('--ng_res', type=int, default=2, help='The number of resblock for generator')
    parser.add_argument('--nd_res', type=int, default=10, help='The number of resblock for discriminator')
    parser.add_argument('--n_mlp', type=int, default=3, help='The number of mlpblokc for decoder')

    parser.add_argument('--img_size', type=int, default=128, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--code_dim', type=int, default=64, help='The dimension of class code')
    parser.add_argument('--n_class', type=int, default=85, help='The number of source classes')
    parser.add_argument('--K', type=int, default=1, help='The number of target class images')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of subprocesses')
    parser.add_argument('--benchmark_flag', type=str2bool, default=True)
    parser.add_argument('--resume', type=str2bool, default=False)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.name, 'model'))
    check_folder(os.path.join(args.result_dir, args.name, 'img'))
    check_folder(os.path.join(args.result_dir, args.name, 'log'))
    check_folder(os.path.join(args.result_dir, args.name, 'test'))

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = FUNIT(args)

    # build graph
    gan.build_model()

    if args.phase == 'train':
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()

