from transformers import T5ForConditionalGeneration
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from rouge import Rouge
from rouge_score import rouge_scorer

dataset = load_dataset("multi_news")
model_nm = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_nm)
model = T5ForConditionalGeneration.from_pretrained(model_nm)

max_input_length = 512
max_target_length = 30

def preprocess(x):
  model_inputs = tokenizer(
      x['document'],
      max_length = max_input_length,
      padding=True,
      truncation=True
  )
  labels = tokenizer(
      x['summary'],
      max_length = max_target_length,
      padding = True,
      truncation=True
  )
  model_inputs['labels'] = labels['input_ids']
  return model_inputs

tok_ds = dataset.map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer,model=model,return_tensors='pt')

tok_ds = tok_ds.remove_columns(['document','summary'])

batch_size = 8

tok_ds.set_format('torch')
train_dataloader = DataLoader(
    tok_ds["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tok_ds["validation"], collate_fn=data_collator, batch_size=batch_size
)

optimizer = AdamW(model.parameters(), lr=2e-5)

def compute_rouge_scores(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(prediction, reference) for prediction, reference in zip(predictions, references)]
    return scores