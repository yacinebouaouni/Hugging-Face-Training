from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_scheduler
import evaluate
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator


# Load Accelerator
accelerator = Accelerator()
print(accelerator.device)
# Load Dataset 
raw_dataset = load_dataset('glue', 'mrpc')

# Load Tokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize Dataset
def sample_tokenize(sample):
    return tokenizer(sample['sentence1'], sample['sentence2'], truncation=True)

tokenized_datasets = raw_dataset.map(sample_tokenize, batched=True)

# Load data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Format Dataset
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Create Data Loaders
train_loader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
valid_loader = DataLoader(tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator)

# Load model, optimizer and scheduler
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

n_epochs = 3
num_training_steps = n_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Accelerate prepare 
train_loader, valid_loader, model, optimizer = accelerator.prepare(train_loader, valid_loader, model, optimizer)

# Load Evaluation Metric
metric = evaluate.load("glue", "mrpc")

# Train Loop
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(n_epochs):

    # Training
    
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions = predictions, references= batch['labels'])

        score = metric.compute()
        print(score)


