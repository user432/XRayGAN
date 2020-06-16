from utils.FID import *
from CheXNet.model import *
from CheXNet.read_data import ChestXrayDataSet_Sia
from utils.IS import *
from utils.SSIM import *
import torch
from models.Siamese import Classifinet
from collections import OrderedDict
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path1', type=str, nargs=2,default='save_image/original',
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--path2', type=str, nargs=2,default='save_image/OPENI_biplane_batchfirst',
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=12,
                    help='Batch size to use')
parser.add_argument('--Sia-resume', type=str, default='checkpoint/OPENI/SIA/Sia_checkpoint.pth')
parser.add_argument('-c', '--gpu', default='0,1', type=str,
                    help='GPU to use (leave blank for CPU only)')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpus = [int(ix) for ix in args.gpu.split(',')]
CKPT_PATH = './CheXNet/model_new.pth.tar'
NEW_CKPT_PATH = './CheXNet/model_new.pth.tar'

def define_model(args):
    cudnn.benchmark = True
    # print()

    model = DenseNet121(N_CLASSES).to(device)
    model = torch.nn.DataParallel(model).cuda()
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
    return model

def convert(in_file, out_file):
    """Convert keys in checkpoints.

    There can be some breaking changes during the development of mmdetection,
    and this tool is used for upgrading checkpoints trained with old versions
    to the latest one.
    """
    checkpoint = torch.load(in_file)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()
    for key, val in in_state_dict.items():
        print(key)
        if key.startswith('module.densenet121.classifier.0.'):
            new_key = key
            out_state_dict[new_key] = val
        else:
            new_key = key.replace('.0.','0.')
            new_key = new_key.replace('.1.','1.')
            new_key = new_key.replace('.2.','2.')
            out_state_dict[new_key] = val
    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)

def mainIS(args, model):


    # print(model)
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset2 = ChestXrayDataSet(data_dir=args.path2,
                                     transform=transforms.Compose([
                                         transforms.Resize(224),
                                         transforms.ToTensor(),
                                         normalize
                                     ]),
                                     only_f=True)


    test_loader2 = DataLoader(dataset=test_dataset2, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=8, pin_memory=True)


    score_mean,score_std = compute_IS(model,test_loader2)
    print('IS: {} \t std: {} '.format(score_mean,score_std))
    return score_mean,score_std

def mainFID(args, model):

    # print(model)
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset1 = ChestXrayDataSet(data_dir=args.path1,
                                    transform=transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ]),
                                     only_f=True)
    test_dataset2 = ChestXrayDataSet(data_dir=args.path2,
                                     transform=transforms.Compose([
                                         transforms.Resize(224),
                                         transforms.ToTensor(),
                                         normalize
                                    ]),
                                     only_f=True)

    test_loader1 = DataLoader(dataset=test_dataset1, batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)
    test_loader2 = DataLoader(dataset=test_dataset2, batch_size=args.batch_size,
                              shuffle=False, num_workers=8, pin_memory=True)
    test_loader = [test_loader1,test_loader2]

    fid_value = calculate_fid_given_paths(model.module.densenet121.features,
                                          test_loader,
                                          device)
    print('FID: ', fid_value)
    return fid_value

def mainSSIM(args, model):



    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet_paired(data_dir1=args.path1,data_dir2 = args.path2,
                                    transform=transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ]),
                                     only_f=True)


    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)

    print("Generating features for {}".format(test_loader.dataset.data_dir1))
    print("Generating features for {}".format(test_loader.dataset.data_dir2))
    ss = []
    for i, (img1,img2) in enumerate(tqdm(test_loader)):
        s = ssim(img1, img2)
        ss.append(s)
    SSIM_value = np.mean(ss)
    print('SSIM: ', SSIM_value)

def mainMS(args):
    embednet = Classifinet(backbone='resnet18')
    embednet = nn.DataParallel(embednet).to(device)
    embednet.load_state_dict(torch.load(args.Sia_resume))
    print("model loaded {}".format(args.Sia_resume))

    test_dataset = ChestXrayDataSet_Sia(data_dir=args.path2,
                                           transform=transforms.Compose([
                                               Rescale((256,256)),
                                               # Equalize(),
                                               ToTensor()
                                           ]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=16,
                             shuffle=False, num_workers=8, pin_memory=True)

    print("Generating features for {}".format(test_loader.dataset.data_dir))

    score = []
    embednet.eval()
    with torch.no_grad():
        for i, (imgf, imgl) in enumerate(tqdm(test_loader)):

            pred = embednet(imgf.to(device), imgl.to(device))
            score.append(pred.data.cpu().numpy())
        score = np.concatenate(score, axis=0)
        print('Maching score: ', np.mean(score))
if __name__ == '__main__':
    # convert(CKPT_PATH,NEW_CKPT_PATH)
    # model = define_model(args)
    # mainSSIM(args, model)
    # mainIS(args, model)
    # mainFID(args, model)
    mainMS(args)