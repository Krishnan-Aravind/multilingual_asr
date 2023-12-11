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


device='cuda'

parser = argparse.ArgumentParser()

parser.add_argument('--language', type=str, default='French')
parser.add_argument('--manualSeed', type=int, default=123, help='random seed for reproducibility')

args = parser.parse_args()
#Making the language into title case, extracting language ID
args.language=args.language.title()



wandb.init(project="Globalphone_Gender_Normalized", entity="krishnan-aravind",name=args.language)
wandb.config.update(args)

manualSeed=args.manualSeed
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True

args.language_id=args.language[:2].upper()


''' Making a list of speakers, splitting it into train/val/test  '''

speaker_list=os.listdir(f'corpora/Globalphone/{args.language}/wav/')
random.shuffle(speaker_list)
train_spk,val_spk,test_spk=np.split(speaker_list,[int(0.7*len(speaker_list)),int(0.8*len(speaker_list))])


''' Making datasets for each speaker list '''

#Initlaizing Datareader Class
gp_datareader=Globaphone_DataLoader(language=args.language,root='/data/users/akrishnan/multilingual_asr/corpora/Globalphone/')

#Passing lists of speakers to the datareader to read and return the xlsr embeddings
globalphone_train=gp_datareader.generate_dataset_from_speaker_list(train_spk)
globalphone_test=gp_datareader.generate_dataset_from_speaker_list(test_spk)
globalphone_val = gp_datareader.generate_dataset_from_speaker_list(val_spk)



'''   Normalization '''

#Finding mean and STD with the train corpus
x=torch.tensor(globalphone_train['raw_embeddings'])
std_per_layer,mean_per_layer=torch.std_mean(x,axis=0,keepdim=True)

#Standardizing each dataset for 0 mean and unit variance

##Standardizing Train
y=(torch.tensor(globalphone_train['raw_embeddings'])-mean_per_layer)/std_per_layer
globalphone_train=globalphone_train.add_column('input_values',y.tolist())
globalphone_train=globalphone_train.remove_columns('raw_embeddings')

##Standardizing Test
y=(torch.tensor(globalphone_test['raw_embeddings'])-mean_per_layer)/std_per_layer
globalphone_test=globalphone_test.add_column('input_values',y.tolist())
globalphone_test=globalphone_test.remove_columns('raw_embeddings')


##Standardizing Val
y=(torch.tensor(globalphone_val['raw_embeddings'])-mean_per_layer)/std_per_layer
globalphone_val=globalphone_val.add_column('input_values',y.tolist())
globalphone_val=globalphone_val.remove_columns('raw_embeddings')

##Sanity check for standardization with train
x=torch.tensor(globalphone_train['input_values'])
assert torch.sum(torch.round(torch.std_mean(x,axis=0,keepdim=True)[1],decimals=4)) == 0, 'Standarization Mean not correct'
assert torch.sum(1- torch.round(torch.std_mean(x,axis=0,keepdim=True)[0],decimals=4)) == 0, 'Standarization STD not correct'

print('\n\nFinished Normalizing. Now Saving\n\n')
globalphone_train.save_to_disk(f'corpora/Globalphone/{args.language}/globalphone_train_{args.language_id}_normalized.hf')
globalphone_val.save_to_disk(f'corpora/Globalphone/{args.language}/globalphone_val_{args.language_id}_normalized.hf')
globalphone_test.save_to_disk(f'corpora/Globalphone/{args.language}/globalphone_test_{args.language_id}_normalized.hf')

