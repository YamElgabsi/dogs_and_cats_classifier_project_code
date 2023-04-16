import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from PIL import Image
from torchvision.models import VGG16_Weights

size_crop = 224
size_resize = 256
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
category_label_to_name = {'1': 'dog', '2': 'cat', '3': 'horse'}


def load_checkpoint(file_path):
    checkpoint = torch.load(file_path,map_location=torch.device('cpu'))
    model = getattr(torchvision.models, checkpoint['network'])(weights=VGG16_Weights.DEFAULT)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(pil_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img_loader = transforms.Compose([transforms.Resize(size_resize),
                                     transforms.CenterCrop(size_crop),
                                     transforms.ToTensor(),
                                     transforms.Normalize(normalize_mean, normalize_std)
                                     ])
    pil_image = img_loader(pil_image).float()
    np_image = np.array(pil_image)

    return np_image

def predict(pil_image, model, top_k_probabilities=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use GPU if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    model.to(device)
    model.eval()

    np_image = process_image(pil_image)
    tensor_image = torch.from_numpy(np_image)

    inputs = Variable(tensor_image.float())

    if torch.cuda.is_available():
        inputs = Variable(tensor_image.float().cuda())

    inputs = inputs.unsqueeze(dim=0)
    log_probabilities = model.forward(inputs)
    probabilities = torch.exp(log_probabilities)

    top_probabilities, top_classes = probabilities.topk(top_k_probabilities, dim=1)

    class_to_idx_inverted = {model.class_to_idx[c]: c for c in model.class_to_idx}
    top_mapped_classes = list()

    for label in top_classes.cpu().detach().numpy()[0]:
        top_mapped_classes.append(class_to_idx_inverted[label])

    return top_probabilities.cpu().detach().numpy()[0], top_mapped_classes

def get_prediction_graph(path):

        pil_image = Image.open(path).convert('RGB')
        plt.imshow(pil_image)
        #
        top_probabilities, top_classes = predict(pil_image, model, top_k_probabilities=3)
        fig = plt.figure(figsize=(6, 6))
        ax1 = plt.subplot2grid((15, 9), (0, 0), colspan=9, rowspan=9)
        ax2 = plt.subplot2grid((15, 9), (9, 2), colspan=5, rowspan=5)

        ax1.axis('off')
        ax1.imshow(pil_image)

        labels = []
        for c in top_classes:
            labels.append(category_label_to_name[c])

        y_pos = np.arange(3)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Probability')
        ax2.invert_yaxis()
        ax2.barh(y_pos, top_probabilities, xerr=0, align='center', color='blue')
        return fig


# loading model
# model = load_checkpoint('untrained_model.pth')
model = load_checkpoint('checkpoint_2023-02-03 12_32.pth')