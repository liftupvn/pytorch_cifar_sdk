from bluetraining import MLOps
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--name', default='Resnet 18', type=str, help='name of model')
parser.add_argument('--username', default='seta dev', type=str, help='name of user')
parser.add_argument('--ckpt_id', default=None, type=str, help='ckpt_ids list')
parser.add_argument('--dataset_test', default=None, type=str, help='dataset_tests')

parser.add_argument('--accuracy', default=None, type=float, help='accuracy')
parser.add_argument('--f1', default=None, type=float, help='f1')
parser.add_argument('--precision', default=None, type=float, help='precision')

args = parser.parse_args()


MLOps.init('testing', name=args.name, username=args.username)


# Run testing and log results

benchmark = dict()
if args.accuracy is not None:
    benchmark['accuracy'] = args.accuracy
if args.f1 is not None:
    benchmark['f1'] = args.f1
if args.precision is not None:
    benchmark['precision'] = args.precision

print('benchmark: ', benchmark)

# write test sumary
MLOps.log.add_test_results(
    args.ckpt_id,
    args.dataset_test,
    benchmark,
    name=f'test model on {args.dataset_test}'
    )