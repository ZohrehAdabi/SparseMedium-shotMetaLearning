import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

train_people = ['DennisPNoGlassesGrey','JohnGrey','SimonBGrey','SeanGGrey','DanJGrey','AdamBGrey','JackGrey','RichardHGrey',
            'YongminYGrey','TomKGrey','PaulVGrey','DennisPGrey','CarlaBGrey','JamieSGrey','KateSGrey','DerekCGrey',
            'KatherineWGrey','ColinPGrey','SueWGrey','GrahamWGrey','KrystynaNGrey','SeanGNoGlassesGrey','KeithCGrey',
            'HeatherLGrey']
test_people  = ['RichardBGrey','TasosHGrey','SarahLGrey','AndreeaVGrey','YogeshRGrey']
#  ['AdamBTestGrey', 'DerekCTestGrey', 'JamieSTestGrey', 'RichardBTestGrey', 'RichardHTestGrey'] #'JeffNGrey', 'JonGrey'
def num_to_str(num):
    str_ = ''
    if num == 0:
        str_ = '000'
    elif num < 100:
        str_ = '0' + str(int(num))
    else:
        str_ = str(int(num))
    return str_

def get_person_at_curve(person, curve, prefix='filelists/QMUL/images/'):
    faces   = []
    targets = []

    train_transforms = transforms.Compose([transforms.ToTensor()])
    for pitch, angle in curve:
        fname  = prefix + person + '/' + person[:-4] + '_' + num_to_str(pitch) + '_' + num_to_str(angle) +'.jpg'
        img    = Image.open(fname).convert('RGB')
        img    = train_transforms(img)

        faces.append(img)
        pitch_norm = 2 * ((pitch - 60) /  (120 - 60)) -1
        angle_norm = 2 * ((angle - 0)  / (180 - 0)) -1
        targets.append(torch.Tensor([pitch_norm]))

        # plt.imshow(transforms.ToPILImage()(img).convert("RGB"))
        # plt.show()
    faces   = torch.stack(faces)
    targets = torch.stack(targets).squeeze()
    return faces, targets

def get_batch(train_people=train_people, num_samples=108):
    ## generate trajectory
    # amp   = np.random.uniform(-4, 4) #-3, 3
    # phase = np.random.uniform(-5, 5)
    # wave  = [(amp * np.sin(phase + x)) for x in range(num_samples)]
    ## map trajectory to angles/pitches
    angles_base  = list(range(0, 18)) + list(range(18, 0, -1))
    angles = [] 
    for _ in range(3):
        angles.extend(angles_base)
    angles  = [x * 10 for x in angles]
    # pitches = [int(round(((y+3)*10 )+60,-1)) for y in wave]
    # tilt = np.arange(60, 130, step=10)
    # pitches = np.random.choice(tilt, size=num_samples, replace=True)
    pitches = []
    tilt = np.arange(60, 130, step=10)
    p = np.random.choice(tilt, size=6, replace=False)
    for e in p:
        pitches.extend([e]*18)
    curve   = [(p,a) for p, a in zip(pitches, angles)]
    # print(f'pitches:{pitches}')
    # print(f'curve:{curve}')
    inputs  = []
    targets = []
    for person in train_people:
        inps, targs = get_person_at_curve(person, curve)
        inputs.append(inps)
        targets.append(targs)

    return torch.stack(inputs), torch.stack(targets)

def get_unnormalized_label(pitch_norm):

    pitch = ((pitch_norm + 1) *(130-50) / 2) + 50
    pitch = ((pitch_norm + 1) *(120-60) / 2) + 60
    return pitch

def grt_normalized_label(pitch):

    # pitch_norm = 2 * ((pitch - 60) /  (120 - 60)) -1
    pitch_norm = 2 * ((pitch - 50) /  (130 - 50)) -1
    return pitch_norm


        
