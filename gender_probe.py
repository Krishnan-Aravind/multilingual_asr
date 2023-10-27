from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2CTCTokenizer
import numpy as np
import pandas as pd
import tqdm
device='cuda'

common_voice_train=torch.load('common_voice_train_gender.pt')
common_voice_test=torch.load('common_voice_test_gender.pt')


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)




@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor([feature["labels"] for feature in features])

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

import torch.nn.functional as F
import torch.nn as nn


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
    
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
      
        attention_weight:
            att_w : size (N, T, 1)
    
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class CustomXLSRModel(nn.Module):
    def __init__(self):
        super(CustomXLSRModel, self).__init__()
        self.xlsr = Wav2Vec2Model.from_pretrained(
                        "facebook/wav2vec2-xls-r-300m",
                        output_hidden_states=False
                    )
        
        # FREEZING XLSR TO AVOID TRAINING IT
        for param in self.xlsr.parameters():
            param.requires_grad = False

        self.hidden_size=self.xlsr.config.hidden_size
        ### New layers:
        self.linear1 = nn.Linear(self.hidden_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, 1) ## 2 is the number of classes in this example
        self.attention_pooling=SelfAttentionPooling(self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        
    

    def forward(self, input_batch):
        output=self.xlsr(**input_batch)
        raw_embeddings=output['last_hidden_state']

        # raw_embeddings has the following shape: (batch_size, sequence_length, 1024)
        pooled_output = self.attention_pooling(raw_embeddings) ## Attention pooling
        
        #Change this in the future to make it like SAMU XLSR. THe linear layer should project with a tanh activation
        
        linear1_output = F.relu(self.linear1(pooled_output)) #Linear layer with Relu Activation
        linear2_output = F.relu(self.linear2(linear1_output)) #Linear layer with Relu Activation
        linear3_output = F.relu(self.linear3(linear2_output)) #Linear layer with Relu Activation

        linear4_output = self.linear4(linear3_output)
        return self.sigmoid(linear4_output)
    
model=CustomXLSRModel()
model=model.to(device)

from sklearn.model_selection import train_test_split
import random
labels=np.array(common_voice_train['labels'])
female_pos=(np.where(labels==1))[0]
male_pos=np.where(labels==0)[0]
np.random.shuffle(male_pos)
#subsampling equal number of male label positions for stratification
male_pos_subset=male_pos[:len(female_pos)]

#joining indices
total_pos=np.sort(np.concatenate((female_pos,male_pos_subset))).tolist()

training_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(common_voice_train, total_pos), batch_size=4, shuffle=False,collate_fn=data_collator)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

#MAKING SUBSET OF TRAIN DATA, TO CHECK TRAINING LOSS AND SO ON


def train_one_epoch(epoch_index):
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



# Initializing in a separate cell so we can easily add more epochs to the same run

epoch_number = 0

EPOCHS = 20

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)
    print(f'Averaged Loss after EPOCH {epoch}:{avg_loss}')
    epoch_number += 1
    
torch.save(model,'gender_model.pt')