import pandas as pd
import os
import re
import tqdm
import librosa
from datasets import Dataset, load_metric
import json
import IPython.display as ipd
import numpy as np
import random
import torch
from src.data_util import DataCollatorCTCWithPadding
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import wandb
from datasets import load_from_disk


''' PARAMETERS '''
manualSeed=123
BATCH_SIZE=16
root_dir = '/data/users/akrishnan/multilingual_asr/corpora/l2arctic'
model_path="facebook/wav2vec2-base-100h"

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True



wandb.init(project="L2_Arctic Train", entity="krishnan-aravind",name=f"{model_path.split('/')[-1]}")


''' LOADING MODEL AND PROCESSOR '''
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.freeze_feature_extractor()
wandb.log({'model': model.name_or_path.split('/')[-1]})



''' LOADING DATA '''
##SEE DOWN FOR CODE USED TO GENERATE DATA. Now loading saved stuff
l2_data=load_from_disk("corpora/l2arctic/l2_arctic_processed_wav2vec.hf")




''' TRAINING WITH US ACCENTS. Splitting datasets '''

us_indices = np.where(np.array(l2_data['accent'])=='US')[0]
test_indices_accent= np.where(np.array(l2_data['accent'])!='US')[0]


#FOR TRAINING WE SPLIT THE DATA BY UTTERANCES FIRST (EACH SPEAKER DUPLICATES THE DATA)

#SELECTING UTTERANCES FOR SPLITS
US_Data=l2_data.select(us_indices)
utterance_ids=np.unique(US_Data['path'])
np.random.shuffle(utterance_ids)
train_utterances,val_utterances,test_utterances= np.split(utterance_ids,[int(0.7*len(utterance_ids)),int(0.8*len(utterance_ids))])

#COMPILING DATA FOR UTTERANCES
train_indices=np.where((np.isin(l2_data['path'],train_utterances)) & (np.array(l2_data['accent'])=='US'))[0]
val_indices=np.where((np.isin(l2_data['path'],val_utterances)) & (np.array(l2_data['accent'])=='US'))[0]
test_indices_us=np.where((np.isin(l2_data['path'],test_utterances)) & (np.array(l2_data['accent'])=='US'))[0]

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)
np.random.shuffle(test_indices_us)

# np.random.shuffle(indices)
# train_indices,val_indices,test_indices_us= np.split(indices,[int(0.8*len(indices)),int(0.9*len(indices))])


data_collator = DataCollatorCTCWithPadding(processor = processor, padding=True)

training_dataset = l2_data.select(train_indices)

validation_dataset = l2_data.select(val_indices)

test_indices=np.concatenate([test_indices_accent,test_indices_us])
testing_dataset = l2_data.select(test_indices)

np.savez('L2_ARCTIC_DATA_SPLIT_INDICES_WAVE2VEC2.npz',test=test_indices,train=train_indices,val=val_indices)
# l2_data.save_to_disk(f'{root_dir}/l2_arctic_processed_wav2vec.hf')
# l2_data.to_csv(f'{root_dir}/l2_arctic_processed_wav2vec2.csv')




''' Setting UP Trainer '''

wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    #cleaning the predicted str to match the original label preprocessing
#     pred_str_cleaned= re.sub(chars_to_remove_regex, '', pred_str).lower()
    
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer*100}




#Let's calculate the number of warmup steps and other logging steps
num_train_samples=len(training_dataset)
steps_per_epoch=num_train_samples/BATCH_SIZE
warmup_steps=0.25*steps_per_epoch
logging_steps=int(steps_per_epoch/2)

training_args = TrainingArguments(
    output_dir='l2_arctic_train/',
    group_by_length=True,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=50,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=logging_steps,
    eval_steps=logging_steps,
    logging_steps=logging_steps,
    learning_rate=3e-5,
    warmup_steps=warmup_steps,
    save_total_limit=2,
    metric_for_best_model = 'eval_wer',
    greater_is_better=False,
    report_to='wandb',
    seed=manualSeed,
    load_best_model_at_end = True
)


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    tokenizer=processor.feature_extractor,
)

test_accents=np.unique(testing_dataset['accent'])
''' TESTING THE ORIGINAL MODEL '''
for accent in test_accents:
    accent_indices=np.where(np.array(testing_dataset['accent'])==accent)[0]
    accent_test_dataset=testing_dataset.select(accent_indices)
    accent_test_metrics=trainer.evaluate(accent_test_dataset,metric_key_prefix=accent)
    print(accent_test_metrics)
    wandb.log({f'{accent}_wer_Before Train':accent_test_metrics[f'{accent}_wer']})
    

''' TRAINING'''
trainer.train()
trainer.save_model('l2_arctic_xlsr_model_US')


''' TESTING '''
test_accents=np.unique(testing_dataset['accent'])
for accent in test_accents:
    accent_indices=np.where(np.array(testing_dataset['accent'])==accent)[0]
    accent_test_dataset=testing_dataset.select(accent_indices)
    accent_test_metrics=trainer.evaluate(accent_test_dataset,metric_key_prefix=accent)
    print(accent_test_metrics)
    wandb.log({f'{accent}_wer_After_Train':accent_test_metrics[f'{accent}_wer']})
    


    
''' 
*************************CODE USED TO GENERATE DATA******************************


speaker_info = pd.read_csv(f'{root_dir}/speaker_info.csv',sep='|')
speaker_info.columns = ['speaker','gender','accent','wav_files','annotations']


#SUBSAMPLING SPEAKERS FOR DEBUGGING
#speaker_info = speaker_info[speaker_info["speaker"].isin(['YBAA','ZHAA','RMS','SLT','CLB'])]


speaker_files = {}
arctic_data = pd.DataFrame(['path','sentence','name','gender','accent','speaker'])
data_list=[]
for speaker in tqdm.tqdm(speaker_info.itertuples(index=False),total=len(speaker_info)):
    wav_transcript_dict = {}
    for wav in tqdm.tqdm(os.listdir(f'{root_dir}/{speaker.speaker}/wav/')):
        #Getting full paths for the wav transcript pairs
        wav_path = f"{root_dir}/{speaker.speaker}/wav/{wav}"
        transcript_path = f"{root_dir}/{speaker.speaker}/transcript/{wav.replace('.wav','.txt')}"
        
        #Reading transcripts and pairs. Note the forced resampling in librosa.load()
        array,sampling_rate=librosa.load(wav_path,sr=16000)
        transcript=open(transcript_path).read()
        
        data_dict={
            'audio': {'array' : array, 'sampling_rate':sampling_rate},
            'sentence' : transcript,
            'path' : wav,
            'gender' : speaker.gender,
            'accent' : speaker.accent,
            'speaker': speaker.speaker}
        
        data_list.append(data_dict)
        
df=pd.DataFrame(data_list)
#Making Huggingface dataset
l2_data=Dataset.from_pandas(df)




chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch
l2_data = l2_data.map(remove_special_characters)



def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}
vocab = l2_data.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=l2_data.column_names)
vocab_dict = {v: k for k, v in enumerate(sorted(vocab['vocab'][0]))}

#Accounting for blanks and unknowns
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)

#Saving into file
with open(f'{root_dir}/vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
    
    
    



def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch
l2_data = l2_data.map(prepare_dataset, remove_columns=['audio','sentence'])



'''