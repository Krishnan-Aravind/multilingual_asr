from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import torch
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2CTCTokenizer
import numpy as np
import wandb
import pandas as pd
import random
from datasets import Dataset
from scipy.io import wavfile
import tqdm
import argparse
from sklearn.metrics import classification_report
from src.early_stopper import EarlyStopper
from src.model import CustomXLSRModel_mean_pooled
from src.data_util import DataCollatorCTCWithPadding
from datasets import load_from_disk

wandb.init(project="Globalphone_Gender", entity="krishnan-aravind",name='FRENCH')


parser = argparse.ArgumentParser()

parser.add_argument('--layer', type=int, default=24)
parser.add_argument('--language', type=str, default='French')
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

#Making the language into title case, extracting language ID
args.language=args.language.title()
args.language_id=args.language[:2].upper()


globalphone_train=load_from_disk(f'corpora/Globalphone/{args.language}/globalphone_train_{args.language_id}.hf')
globalphone_val=load_from_disk(f'corpora/Globalphone/{args.language}/globalphone_val_{args.language_id}.hf')
globalphone_test=load_from_disk(f'corpora/Globalphone/{args.language}/globalphone_test_{args.language_id}.hf')


print("FINISHED LOADING  DATA")
##################################### Training #################################################



tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

#Datacollator to take care of padding
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    
model=CustomXLSRModel_mean_pooled(layer=args.layer)
model=model.to(device)

training_loader = torch.utils.data.DataLoader(globalphone_train, batch_size=args.batch_size, shuffle=False,collate_fn=data_collator)

validation_loader= torch.utils.data.DataLoader(globalphone_val, batch_size=args.batch_size, shuffle=False,collate_fn=data_collator)

testing_loader=torch.utils.data.DataLoader(globalphone_test, batch_size=args.batch_size, shuffle=False,collate_fn=data_collator)


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

torch.save(model,f'models/globalphone_{args.language_id}_gender_model_layer_{args.layer}.pt')
