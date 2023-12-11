import torch
import os
import numpy as np
import pandas as pd
import random
from scipy.io import wavfile
import tqdm
import argparse
import wandb
from src.make_data_globalphone import Globaphone_DataLoader
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
from src import debias
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import f1_score
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

device='cuda'

parser = argparse.ArgumentParser()

parser.add_argument('--language', type=str, default='French')
parser.add_argument('--manualSeed', type=int, default=123, help='random seed for reproducibility')

args = parser.parse_args()
#Making the language into title case, extracting language ID
args.language=args.language.title()


#wandb.init('offline')
wandb.init(project="Globalphone_Debias_SID", entity="krishnan-aravind",name=f'{args.language}')
wandb.config.update(args)

manualSeed=args.manualSeed
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True

args.language_id=args.language[:2].upper()


globalphone_train=load_from_disk(f'corpora/Globalphone/{args.language}/globalphone_train_{args.language_id}_normalized.hf')
globalphone_val=load_from_disk(f'corpora/Globalphone/{args.language}/globalphone_val_{args.language_id}_normalized.hf')
globalphone_test=load_from_disk(f'corpora/Globalphone/{args.language}/globalphone_test_{args.language_id}_normalized.hf')

X_train_=torch.tensor(globalphone_train['input_values'])
X_val_=torch.tensor(globalphone_val['input_values'])
X_test_=torch.tensor(globalphone_test['input_values'])

''' Making Labels for SPEAKER ID'''

speaker_id=[int(i.split('_')[0][2:]) for i in globalphone_train['utterance_id']]
globalphone_train=globalphone_train.add_column('speaker_id',speaker_id)

speaker_id=[int(i.split('_')[0][2:]) for i in globalphone_val['utterance_id']]
globalphone_val=globalphone_val.add_column('speaker_id',speaker_id)

speaker_id=[int(i.split('_')[0][2:]) for i in globalphone_test['utterance_id']]
globalphone_test=globalphone_test.add_column('speaker_id',speaker_id)


y_train_=globalphone_train['speaker_id']
y_test_=globalphone_test['speaker_id']
y_val_=globalphone_val['speaker_id']

''' STRATIFIED SPLITTING TO BALANCE SPEAKERS FOR SPEAKER_ID'''

X = np.concatenate([X_train_,X_val_,X_test_])
y = np.concatenate([y_train_,y_val_,y_test_])

z= np.concatenate([globalphone_train['labels'],globalphone_val['labels'],globalphone_test['labels']])

#Splitting train and test first
train_indices, test_indices, y_train, y_test = train_test_split(range(X.shape[0]), y,
                                                    stratify=y, 
                                                    test_size=0.2,
                                                   random_state=123)
#Splitting train again to get val
train_indices, val_indices, y_train, y_val  = train_test_split(train_indices, y_train, test_size=0.125, random_state=123,stratify=y_train) # 0.125 x 0.8 = 0.1

###RESTORING OLD SPLIT (SPEAKER INSULATING)
#train_indices, val_indices, test_indices = np.split(range(len(X)),[len(y_train_),len(y_train_)+len(y_val_)])

for layer in [0,1,2,4,8,12,13,14,15,16,17,20,22,23,24]:
    X_train_layer = np.squeeze(X[train_indices,layer,:,:])
    X_test_layer = np.squeeze(X[test_indices,layer,:,:])
    X_val_layer= np.squeeze(X[val_indices,layer,:,:])
    z_train=z[train_indices]
    z_test=z[test_indices]
    z_val=z[val_indices]

    clf = SGDClassifier(random_state=args.manualSeed, max_iter=200).fit(X_train_layer, z_train)
    #This does not make sense since the speakers are mixed anyway. There is some leakage
    wandb.log({'Gender F1_Score Before Debias':f1_score(clf.predict(X_test_layer), z_test)},step=layer)
    clf = SGDClassifier(random_state=args.manualSeed, max_iter=500).fit(X_train_layer, y_train)
    wandb.log({'SID F1_Score Before Debias':f1_score(clf.predict(X_test_layer), y_test,average='macro')},step=layer)  
    
    num_classifiers = 200
    classifier_class = SGDClassifier #Perceptron
    input_dim = X_train_layer.shape[1]
    is_autoregressive = True
    min_accuracy = 0.0

    P, rowspace_projections, Ws = debias.get_debiasing_projection(classifier_class, {}, num_classifiers, input_dim, is_autoregressive, min_accuracy, X_train=X_train_layer, Y_train=z_train, X_dev=X_val_layer, Y_dev=z_val, by_class = False)

    X_train_layer_projected=np.matmul(X_train_layer,P)
    X_test_layer_projected=np.matmul(X_test_layer,P)

    clf = SGDClassifier(random_state=args.manualSeed, max_iter=200).fit(X_train_layer_projected, z_train)
    wandb.log({'Gender F1_Score After Debias':f1_score(clf.predict(X_test_layer_projected), z_test)},step=layer)  
    
    clf = SGDClassifier(random_state=args.manualSeed, max_iter=1000).fit(X_train_layer_projected, y_train)
    wandb.log({'SID F1_Score After Debias':f1_score(clf.predict(X_test_layer_projected), y_test,average='macro')},step=layer)   

    
