import torch
from torchvision import transforms
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from datautils import data_loader
from comq import COMQ
from modelutils import BatchNrom_tuning

def test_accuracy(model, test_dl, topk=(1, )):

    model.eval()
    maxk = max(topk)
    topk_count = np.zeros((len(topk), len(test_dl)))
    
    for j, (x_test, target) in enumerate(tqdm(test_dl)):
        x_test = x_test.to(device)
        target = target.to(device)
        with torch.no_grad():
            y_pred = model(x_test)
        topk_pred = torch.topk(y_pred, maxk, dim=1).indices
        target = target.view(-1, 1).expand_as(topk_pred)
        correct_mat = (target == topk_pred)

        for i, k in enumerate(topk):
            topk_count[i, j] = correct_mat[:, :k].reshape(-1).sum().item()

    topk_accuracy = topk_count.sum(axis=1) / len(test_dl.dataset)
    return topk_accuracy

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='model name'
    )
    parser.add_argument(
        '--data_path', default='', type=str, 
        help='path to ImageNet data'
    )
    parser.add_argument(
        '--batchsize', type=int, default=1024,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--num_workers',
        type=int, default=8, help='number of wrker for loading data.'
    )
    parser.add_argument(
        '--wbits', type=int, default=4, choices=[2, 3, 4],
        help='#bits to use for quantization.'
    )
    parser.add_argument(
        '--greedy', action='store_true',
        help='Whether to apply the order COMQ'
    )
    parser.add_argument(
        '--batchtuning', action='store_true',
        help='Whether to apply the BatchNorm tuning'
    )
    parser.add_argument(
        '--iters', type=int, default=1,
        help='#iterations to use for quantization'
    )
    parser.add_argument(
        '--scalar', default=1.0, type=float,
        help='#scale to revise for bit-code.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # load model
    model = timm.create_model(args.model, pretrained=True)
    model = model.to(device)
    model.eval()

    # build imagenet data loader
    config = resolve_data_config(model.pretrained_cfg)
    train_transforms = create_transform(**config, is_training=True)
    test_transforms = create_transform(**config)
    train_loader, test_loader = data_loader(args.data_path, args.batchsize, train_transforms, test_transforms, args.num_workers)

    print(f'\n Starting to quantize {args.model} for {args.wbits} bits:\n')

    # quantize the model
    quantizer = COMQ(model=model,
                     data_loader=train_loader, 
                     model_name=args.model,
                     bits=args.wbits,
                     iters=args.iters,
                     scalar=args.scalar,
                     greedy=args.greedy,
                     device=device)
        
    start_time = datetime.now()

    quantized_model = quantizer.quantize_model()

    end_time = datetime.now()
    
    print(f'\nTime for quantization: {end_time - start_time}\n')

    if args.batchtuning:
        batch_tuning_loader, test_loader = data_loader(args.data_path, 128, train_transforms, test_transforms, args.num_workers)
        BatchNrom_tuning(quantized_model, args.model, batch_tuning_loader, device)

    print(f'\n Evaluting the quantized model: {args.model} with {args.wbits} bits\n')

    topk_accuracy = test_accuracy(quantized_model, test_loader, (1, 5))

    print(f'Top-1 accuracy is {topk_accuracy[0]}.')
    print(f'Top-5 accuracy is {topk_accuracy[1]}.')

    if args.save:
        torch.save(quantized_model.state_dict(), args.save)
