from tqdm.auto import tqdm
import torch
import numpy as np
from rouge_score import rouge_scorer
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

num_train_epochs = 5
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device

        
        # Forward pass
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        labels=batch['labels'])
        logits = outputs.logits

        # Calculate loss
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
    
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average Loss: {average_loss:.4f}")
      # Evaluation
    model.eval()
    rouge_scores = []
    for batch in eval_dataloader:
        with torch.no_grad():
            batch.to(device)
            outputs = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=50  # Adjust max_length according to your requirements
            )

        # Decode generated summaries and references
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_refs = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        # Compute ROUGE scores
        scores = compute_rouge_scores(decoded_preds, decoded_refs)
        rouge_scores.extend(scores)

    # Calculate average ROUGE scores
    avg_rouge_scores = {
        'rouge1': sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores),
        'rouge2': sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores),
        'rougeL': sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    }

    print("Average ROUGE Scores:", avg_rouge_scores)

    # Adjust learning rate
    lr_scheduler.step()