import argparse

from infrastructure import *

from models.dense_net import DenseNet
from data_providers.utils import get_data_provider_by_name
import sys
import traceback

train_params_cifar = {
    'batch_size': 100,
    'n_epochs': 1,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 100,
    'n_epochs': 20,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}


def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--train', action='store_true',
            help='Train the model')
        parser.add_argument(
            '--test', action='store_true',
            help='Test model for required dataset if pretrained model exists.'
                 'If provided together with `--train` flag testing will be'
                 'performed right after training.')
        parser.add_argument(
            '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
            default='DenseNet',
            help='What type of model to use')
        parser.add_argument(
            '--growth_rate', '-k', type=int, choices=[2, 12, 24, 40],
            default=12,
            help='Growth rate for every layer, '
                 'choices were restricted to used in paper')
        parser.add_argument(
            '--depth', '-d', type=int, choices=[40, 100, 129, 154, 190, 250],
            default=40,
            help='Depth of whole network, restricted to paper choices')
        parser.add_argument(
            '--dataset', '-ds', type=str,
            choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN'],
            default='C10',
            help='What dataset should be used')
        parser.add_argument(
            '--total_blocks', '-tb', type=int, default=3, metavar='',
            help='Total blocks of layers stack (default: %(default)s)')
        parser.add_argument(
            '--keep_prob', '-kp', type=float, metavar='',
            help="Keep probability for dropout.")
        parser.add_argument(
            '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
            help='Weight decay for optimizer (default: %(default)s)')
        parser.add_argument(
            '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
            help='Nesterov momentum (default: %(default)s)')
        parser.add_argument(
            '--reduction', '-red', type=float, default=0.5, metavar='',
            help='reduction Theta at transition layer for DenseNets-BC models')

        parser.add_argument(
            '--logs', dest='should_save_logs', action='store_true',
            help='Write tensorflow logs')
        parser.add_argument(
            '--no-logs', dest='should_save_logs', action='store_false',
            help='Do not write tensorflow logs')
        parser.set_defaults(should_save_logs=True)

        parser.add_argument(
            '--saves', dest='should_save_model', action='store_true',
            help='Save model during training')
        parser.add_argument(
            '--no-saves', dest='should_save_model', action='store_false',
            help='Do not save model during training')
        parser.set_defaults(should_save_model=True)

        parser.add_argument(
            '--renew-logs', dest='renew_logs', action='store_true',
            help='Erase previous logs for model if exists.')
        parser.add_argument(
            '--not-renew-logs', dest='renew_logs', action='store_false',
            help='Do not erase previous logs for model if exists.')
        parser.set_defaults(renew_logs=True)
        parser.add_argument(
            '--sdr', dest='use_sdr', action='store_true',
            help='Use Stochastic Delta Rule instead of dropout.')
        parser.set_defaults(use_sdr=False)

        parser.add_argument('--stochastic',dest='stochastic', action='store_true', help='Run the network with stochastic activations')
        parser.set_defaults(stochastic=False)
        parser.add_argument('--slr', dest='stochastic_learning_rate', action='store_true', help='Run the network with a stochastic learning rate')
        parser.set_defaults(stochastic_learning_rate = False)
        parser.add_argument('--exp_name', type=str, help='The name of the experiment you want')

        args = parser.parse_args()

        if not args.keep_prob:
            if args.dataset in ['C10', 'C100', 'SVHN']:
                args.keep_prob = 0.8
            else:
                args.keep_prob = 1.0

        if args.use_sdr:
            args.keep_prob = 1.0
        if args.model_type == 'DenseNet':
            args.bc_mode = False
            args.reduction = 1.0
        elif args.model_type == 'DenseNet-BC':
            args.bc_mode = True

        model_params = vars(args)

        if not args.train and not args.test:
            print("You should train or test your network. Please check params.")
            exit()

        # some default params dataset/architecture related
        train_params = get_train_params_by_name(args.dataset)
        send_mail("Starting Experiment: " + str(args.exp_name), "Starting experiment: " + datestring() + "\n" + str(args) + "\n" + str(train_params) + "\n")
        print("Params:")
        for k, v in model_params.items():
            print("\t%s: %s" % (k, v))
        print("Train params:")
        for k, v in train_params.items():
            print("\t%s: %s" % (k, v))

        print("Prepare training data...")
        data_provider = get_data_provider_by_name(args.dataset, train_params)
        print("Initialize the model..")
        model = DenseNet(data_provider=data_provider, **model_params)
        if args.train:
            print("Data provider train images: ", data_provider.train.num_examples)
            model.train_all_epochs(train_params)
        if args.test:
            if not args.train:
                model.load_model()
            print("Data provider test images: ", data_provider.test.num_examples)
            print("Testing...")
            loss, accuracy = model.test(data_provider.test, batch_size=200)
            print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))


    except Exception as e:
        # this should hopefully send mail whenever there is a significant issue. Wrapping this whole thing in try-catch
        # should hopefully not cause performance issues!
        print("Exception raised while running network")
        info = type_, value_, traceback_ = sys.exc_info()
        tb = traceback.format_tb(traceback_)
        s = format_traceback_exception_log(e, info, tb)
        send_mail("EXCEPTION: Stochastic Nets " + str(args.exp_name), s)
        log(s + "\n")
    finally:
        log("Shutting down experiment: " + datestring() + "\n")
    


