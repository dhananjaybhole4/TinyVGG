import torch
from PIL import Image
import argparse
from pathlib import Path

from utils.transform import transform

def prediction(model_path,
               transform,
               img_path,
               device):
    img = Image.open(img_path)
    X = transform(img).unsqueeze(dim = 0).to(device)
    model = torch.load(model_path, weights_only = False)
    model.eval()
    with torch.inference_mode():
        Y_pred = model(X)
    classes = ['chicken_curry', 'chocolate_cake', 'ice_cream']
    prediction = classes[torch.argmax(torch.softmax(Y_pred, dim = 1), dim = 1)]
    print(prediction)
    return prediction

def main():
    # device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required = True, type = str)
    parser.add_argument('--model_name', required = True)

    args = parser.parse_args()

    img_path = Path.cwd()/'sample'/(args.image_name + '.jpg')
    model_path = Path.cwd()/'pretrained_model'/(args.model_name + '.pth')

    prediction(model_path,
               transform,
               img_path,
               device)

if __name__ == '__main__':
    main()