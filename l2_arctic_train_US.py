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


''' HYPERPARAMETERS '''
manualSeed=123
BATCH_SIZE=16
root_dir = '/data/users/akrishnan/multilingual_asr/corpora/l2arctic/cmu_us_en/'


random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.deterministic = True



wandb.init(project="L2_Arctic Train", entity="krishnan-aravind",name='US_XLSR_300M')



speaker_info = pd.read_csv(f'{root_dir}/speaker_info.csv',sep='|')
speaker_info.columns = ['speaker','gender','accent']

speaker_files = {}
data_list=[]
for speaker in tqdm.tqdm(speaker_info.itertuples(index=False),total=len(speaker_info)):
   
    transcripts={i[1:-1].strip().split()[0]:" ".join(i[1:-1].strip().split()[1:])[1:-1] for i in open(f'{root_dir}/cmu_us_{speaker.speaker}_arctic/etc/txt.done.data').read().splitlines()}

    wav_transcript_dict = {}
    for wav in tqdm.tqdm_notebook(os.listdir(f'{root_dir}/cmu_us_{speaker.speaker}_arctic/wav/')):
        #Getting full paths for the wav transcript pairs
        wav_path = f"{root_dir}/cmu_us_{speaker.speaker}_arctic/wav/{wav}"
        transcript_id = wav.replace('.wav','').strip()
        
        #Reading transcripts and pairs. Note the forced resampling in librosa.load()
        array,sampling_rate=librosa.load(wav_path,sr=16000)
        try:
            transcript=transcripts[transcript_id]
        except:
            print(f'transcript not found for speaker {speaker.speaker}, wav file {wav}. Skipping..')
            continue
            
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

# #Saving into file
# with open(f'{root_dir}/vocab.json', 'w') as vocab_file:
#     json.dump(vocab_dict, vocab_file)
    
    
    
    
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)



def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch
l2_data = l2_data.map(prepare_dataset, remove_columns=['audio','sentence'])


indices = np.arange(len(l2_data))
np.random.shuffle(indices)
train_indices,val_indices,test_indices = np.split(indices,[int(0.7*len(indices)),int(0.8*len(indices))])


data_collator = DataCollatorCTCWithPadding(processor = processor, padding=True)

training_dataset = l2_data.select(train_indices)

validation_dataset = l2_data.select(val_indices)

testing_dataset = l2_data.select(test_indices)

training_dataset.save_to_disk(f'corpora/Globalphone/{args.language}/globalphone_train_{args.language_id}_normalized.hf')
validation_dataset.save_to_disk(f'corpora/Globalphone/{args.language}/globalphone_val_{args.language_id}_normalized.hf')
testing_dataset.save_to_disk(f'corpora/Globalphone/{args.language}/globalphone_test_{args.language_id}_normalized.hf')




wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer*100}



model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
model.freeze_feature_extractor()



training_args = TrainingArguments(
    output_dir='l2_arctic_train/',
    group_by_length=True,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=100,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
    push_to_hub=False,
    metric_for_best_model = 'eval_wer',
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


trainer.train()
trainer.save_model('models/l2_arctic_xlsr_model_US')