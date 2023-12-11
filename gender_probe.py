from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import torch
import random
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2CTCTokenizer
import numpy as np
import pandas as pd

import tqdm
import argparse
from sklearn.metrics import classification_report
from src.early_stopper import EarlyStopper
from src.model import CustomXLSRModel_mean_pooled
from src.data_util import DataCollatorCTCWithPadding

import wandb


wandb.init(project="speechprobe_mean_pooled", entity="krishnan-aravind",name='Gender_ID')


parser = argparse.ArgumentParser()

parser.add_argument('--layer', type=int, default=24)

parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--manualSeed', type=int, default=47, help='random seed for reproducibility')
args = parser.parse_args()
wandb.config.update(args)


manualSeed=args.manualSeed
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True

device='cuda'

common_voice_train=torch.load('corpora/common_voice_turkish/common_voice_train_gender.pt')
common_voice_test=torch.load('corpora/common_voice_turkish/common_voice_test_gender.pt')


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)




data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    
model=CustomXLSRModel_mean_pooled(layer=args.layer)
model=model.to(device)

from sklearn.model_selection import train_test_split
import random
labels=np.array(common_voice_train['labels'])
female_pos=(np.where(labels==1))[0]
male_pos=np.where(labels==0)[0]
np.random.shuffle(male_pos)
np.random.shuffle(female_pos)
#subsampling equal number of male label positions for stratification
male_pos_subset=male_pos[:len(female_pos)]

male_train,male_val=np.split(male_pos_subset,[int(0.8*len(male_pos_subset))])
female_train,female_val=np.split(female_pos,[int(0.8*len(female_pos))])
#joining indices
train_pos=np.sort(np.concatenate((male_train,female_train))).tolist()
val_pos=np.sort(np.concatenate((male_val,female_val))).tolist()

training_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(common_voice_train, train_pos), batch_size=args.batch_size, shuffle=False,collate_fn=data_collator)

validation_loader= torch.utils.data.DataLoader(torch.utils.data.Subset(common_voice_train, val_pos), batch_size=args.batch_size, shuffle=False,collate_fn=data_collator)

testing_loader=torch.utils.data.DataLoader(common_voice_test, batch_size=args.batch_size, shuffle=False,collate_fn=data_collator)


loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)


def train_one_epoch():
    running_loss = 0.
    average_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, inputs in tqdm.tqdm(enumerate(training_loader),total=len(training_loader)):
        # Every data instance is an input + label pair
        inputs=inputs.to(device)
        labels = inputs.pop('labels')
        labels=labels.reshape(len(labels),-1).type(torch.float32)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather loss across all batches
        running_loss += loss.item()
        

    average_loss = running_loss / (i+1) # loss per batch

    return average_loss


def validate():
    #set model to eval mode
    model.eval()
    pred_labels=[]
    true_labels=[]
    with torch.no_grad():
        for i, inputs in tqdm.tqdm(enumerate(validation_loader),total=len(validation_loader)):
            inputs=inputs.to(device)
            labels = inputs.pop('labels')
            labels=labels.reshape(len(labels),-1).type(torch.float32)
            outputs = model(inputs)
            true_labels.extend(torch.flatten(labels).tolist())
            pred_labels.extend(torch.round(torch.flatten(outputs)).tolist())
    print(classification_report(y_pred=pred_labels,y_true=true_labels,output_dict=True))
    acc=np.sum(np.equal(pred_labels,true_labels))/len(pred_labels)
    return acc

def test():
    #set model to eval mode
    model.eval()
    pred_labels=[]
    true_labels=[]
    with torch.no_grad():
        for i, inputs in tqdm.tqdm(enumerate(testing_loader),total=len(testing_loader)):
            inputs=inputs.to(device)
            labels = inputs.pop('labels')
            labels=labels.reshape(len(labels),-1).type(torch.float32)
            outputs = model(inputs)
            true_labels.extend(torch.flatten(labels).tolist())
            pred_labels.extend(torch.round(torch.flatten(outputs)).tolist())
    print(classification_report(y_pred=pred_labels,y_true=true_labels,output_dict=True))
    test_op=pd.DataFrame(classification_report(y_pred=pred_labels,y_true=true_labels,output_dict=True))
    return test_op


# Initializing in a separate cell so we can easily add more epochs to the same run

epoch_number = 0
EPOCHS = 500

early_stopper = EarlyStopper(patience=args.patience, delta=0.03,
                             large_is_better=True,
                             save_dir=None,
                             verbose=False,
                             trace_func=print)

for epoch in range(EPOCHS):
    
    print('EPOCH {}:'.format(epoch + 1))
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch()
    print(f'Averaged Loss after EPOCH {epoch}:{avg_loss}')
    wandb.log({'train_loss':avg_loss})
    
    if epoch%2==0:
        val_acc=validate()
        print(f'Validation Accuracy after EPOCH {epoch}:{val_acc}')
        wandb.log({'val_accuracy':val_acc})
        
        #Check if earlystopping is to be triggered
        early_stopper.register(val_acc, model,
                               optimizer, current_step=epoch)
        if early_stopper.early_stop:   
            print(f'\n\n\nEarly stopping triggered at epoch {epoch}\n\n\n')
            break

#Reloading the best model
model = CustomXLSRModel_mean_pooled(layer=args.layer)
model.load_state_dict(early_stopper.get_final_res()['es_best_model'])
model=model.to(device)
optimizer.load_state_dict(early_stopper.get_final_res()['es_best_opt'])


#Testing

print(f"Loaded best model with Validation accuracy {early_stopper.get_final_res()['best_score']}")
print('\n\n\nValidation Results for this model (to cross check if correct model is loaded)')
validate()
test_metrics=test()
wandb.log({'test_f1_avg':test_metrics['macro avg']['f1-score']})
test_wandb=wandb.Table(dataframe=test_metrics)

wandb.log({'test':test_wandb,'layer':args.layer})

torch.save(model,f'models/gender_model_layer_{args.layer}.pt')
