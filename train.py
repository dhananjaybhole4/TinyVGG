from TinyVGG.dataset_preprocessing import dataloader
from TinyVGG.model import ImageClassifier
from TinyVGG.training_loop import train
from utils.save_model import save_model
from utils.plot_loss_curve import plot_loss_curve
from utils.transform import transform
from utils.load_pretrained_model import load_pretrained_model

from pathlib import Path
import os
import argparse

import torch
import torch.nn as nn

training_path = Path.cwd()/'dataset'/'custom_dataset'/'training_dataset'
testing_path = Path.cwd()/'dataset'/'custom_dataset'/'testing_dataset'

NUM_WORKERS = os.cpu_count()

# Set manual seed
torch.manual_seed(42)

# model instance
Model_0 = ImageClassifier(input_layers = 3,
                          hidden_layers = 10,
                          output_layers = 3)


# loss function
loss_fn = nn.CrossEntropyLoss()

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # create parser
    parser = argparse.ArgumentParser()

    # Adding arguments
    parser.add_argument("--name", required = True, type = str, help = 'name of the model to be saved')
    parser.add_argument("--use_pretrained_model", default = False, type = bool, help = 'way to choose a model from sratch or pretrained complex model like efficientnet')
    parser.add_argument("--lr", default = 0.001, type = float, help = 'learning rate')
    parser.add_argument("--epochs", default = 5, type = int, help = 'epochs to train the model')
    parser.add_argument("--batch_size", default = 32, type = int, help = 'batch_size for a dataset')
    parser.add_argument("--loss_curve", required = False, default = False, type = bool, help = 'getting a loss curve')

    # read the argument
    args = parser.parse_args()


    if not args.use_pretrained_model:
        # Setting manual seed
        torch.manual_seed(42)
        # optimizer
        optimizer = torch.optim.Adam(Model_0.parameters(), lr = args.lr)

        # Getting Dataloader (dataset_path --> dataset --> dataloader)
        training_dataloader = dataloader(data_path = training_path,
                                        batch_size = args.batch_size,
                                        shuffle = True,
                                        num_workers = NUM_WORKERS,
                                        transform = transform)
        testing_dataloader = dataloader(data_path = testing_path,
                                        batch_size = args.batch_size,
                                        shuffle = False,
                                        num_workers = NUM_WORKERS,
                                        transform = transform)
        
        print(device)

        # training the model
        epoch_count, train_loss_count, train_acc_count, test_loss_count, test_acc_count = train(epochs = args.epochs,
                                                                                                model = Model_0,
                                                                                                train_dataloader = training_dataloader,
                                                                                                test_dataloader = testing_dataloader,
                                                                                                loss_fn = loss_fn,
                                                                                                optimizer = optimizer,
                                                                                                device = device)
        # saved model path
        path = Path.cwd()/'pretrained_model'/(args.name + str(args.epochs) + '.pth')

        # Saving the model
        save_model(model = Model_0,
                path = path,
                full_model = True)
        
        # plotting the loss curve
        if args.loss_curve:
            plot_loss_curve(train_loss_count, test_loss_count)

    else:
        # loading model
        pre_trained_model, new_transform = load_pretrained_model()

        # Setting manual seed
        torch.manual_seed(42)

        # optimizer
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr = args.lr)

        # Getting Dataloader (dataset_path --> dataset --> dataloader)
        training_dataloader = dataloader(data_path = training_path,
                                        batch_size = args.batch_size,
                                        shuffle = True,
                                        num_workers = NUM_WORKERS,
                                        transform = new_transform)
        testing_dataloader = dataloader(data_path = testing_path,
                                        batch_size = args.batch_size,
                                        shuffle = False,
                                        num_workers = NUM_WORKERS,
                                        transform = new_transform)
        
        print(device)

        # training the model
        epoch_count, train_loss_count, train_acc_count, test_loss_count, test_acc_count = train(epochs = args.epochs,
                                                                                                model = pre_trained_model,
                                                                                                train_dataloader = training_dataloader,
                                                                                                test_dataloader = testing_dataloader,
                                                                                                loss_fn = loss_fn,
                                                                                                optimizer = optimizer,
                                                                                                device = device)
        # saved model path
        path = Path.cwd()/'pretrained_model'/(args.name + str(args.epochs) + '.pth')

        # Saving the model
        save_model(model = pre_trained_model,
                path = path,
                full_model = True)
        
        # plotting the loss curve
        if args.loss_curve:
            plot_loss_curve(train_loss_count, test_loss_count)



if __name__ == '__main__':
    main()
