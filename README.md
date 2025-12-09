# TinyVGG

 This repository contains a neural network model inspired by TinyVGG. It include full data preparation containing three classes ['chicken_curry','chocolate_cake','ice_cream'] , model building, training and evaluation forming a complete machine learning pipeline

 ### Image

Following image represent the structure of neural network model. The only difference is, there are three classes or output terms in the model built in the repository

![TinyVGG](./images/a.png)

## Installation instruction for users

- Clone the repository
    `git clone https://github.com/Dhananjay-AD/TinyVGG.git`
- check for requirements
    `pip install -r requirements.txt`
- place the custom dataset in the following way
    - TinyVGG/dataset/custom dataset/
    - training dataset/
        - class1/
        - class2/
        - class3/
    - testing dataset/
        - class1/
        - class2/
        - class3/

Replace `class1`, `class2`, `class3` with your own class names

## Training the neural network
Run `train.py` with arguments 
- `--name` - set the name of the model
- `--use_pretrained_model` - train with the efficient_net model with complex structure and pre_trained weight
- `--lr` - set learning rate
- `--epochs` -  set number of epochs for training
- `--batch_size` - set batch size
- `--loss_curve` - set True to visualize the loss curve

example - `python3 train.py --name model0 --lr 0.001 --epochs 20 --batch_size 32 --loss_curve True`