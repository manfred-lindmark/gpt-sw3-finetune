import os
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import prepare_model_for_kbit_training
from peft import PeftModel, LoraConfig, get_peft_model
import bitsandbytes as bnb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# See https://huggingface.co/AI-Sweden-Models/ for details and license of
# the GPT-SW3 models.

# Link to Huggingface model to get tokenizer and config
base_model = "AI-Sweden-Models/gpt-sw3-6.7b-v2"

# base_model_checkpoint is either same as base_model to download the model
# from huggingface, or the path to a local model.
# You can save some disk space by saving a half precision (bfloat16) copy
# of the model (using model.save_pretrained(), example in generated_text.py)
base_model_checkpoint = "base-models/gpt-sw3-6.7b-v2-bf16"

# Set to the path of a saved checkpoint or set to False to start training a
# new LORA model.
resume_from = False

# An example Swedish text corpus can be downloaded by running get_dataset.py
# This is a collection of horror stories told on the podcast "Creepypodden".
# https://www.creepypasta.se/
train_data_file = "datasets/corpus_train.txt"

# Context is the length of the models "memory", best to keep at 2048 for GPT2
# models, but can be reduced to save memory when training.
CONTEXT = 2048

batch_size = 2
gradient_accumulation_steps = 2
effective_batch_size = batch_size * gradient_accumulation_steps
epochs = 2
max_train_steps = 10000


# Separate validation data, text that is not part of the training set.
with open('datasets/corpus_validation.txt', encoding='utf-8') as f:
    VALIDATION_TEXT = f.read()


def eval_model(model, tokenizer, generate=False):
    set_seed(42)
    input_ids = tokenizer(VALIDATION_TEXT, return_tensors="pt")["input_ids"][0].to(device)
    splits = [torch.unsqueeze(input_ids[i-CONTEXT:i], 0) for i in range(CONTEXT, len(input_ids), CONTEXT)]

    assert not any(len(c[0]) > CONTEXT for c in splits)
    if generate:
        with torch.no_grad():
            print('Generating text')
            generated_token_ids = model.generate(
                inputs=splits[0][:,:CONTEXT-300],
                max_new_tokens=250,
                do_sample=True,
                temperature=0.50,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.10,
            )[0]
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(generated_text[-1000:])
    
    eval_loss = 0

    for split in splits:        
        attention_mask = torch.ones_like(split)
        labels = split
        with torch.no_grad():
            outputs = model(split, attention_mask=attention_mask, labels=labels)
        eval_loss += outputs.loss.detach().float()   
    eval_loss /= len(splits)
    print(f' Eval loss: {eval_loss:.3f}')
    

with open(train_data_file, encoding='utf-8') as f:
    train_text = f.read()    

tokenizer = AutoTokenizer.from_pretrained(base_model,
                                          model_max_length=1000000000,
                                          padding_side="left",
                                          add_eos_token=False,
                                          add_bos_token=False,
                                          )
tokenizer.pad_token = tokenizer.eos_token


train_tokens = tokenizer(train_text, return_tensors="pt")['input_ids'][0]
print("Number of train tokens:", len(train_tokens))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(base_model_checkpoint,
                                             quantization_config=bnb_config,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")


#model.gradient_checkpointing_enable() # Save memory at the cost of training speed
base_model = prepare_model_for_kbit_training(base_model)

config = LoraConfig(
    r=128,
    lora_alpha=32,
    target_modules=[
        "wpe",
        "c_fc",
        "c_attn",
        "c_proj",
        "lm_head",
    ],
    bias="none",
    fan_in_fan_out=True,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

if not resume_from:
    model = get_peft_model(base_model, config)
else:
    print('Resuming training from QLORA:', resume_from)
    model = PeftModel.from_pretrained(base_model, resume_from, is_trainable=True, config=config)

eval_model(model, tokenizer, generate=True)

optimizer = bnb.optim.PagedLion8bit(model.parameters(), lr=2e-5, betas=(0.95, 0.98), 
                        weight_decay=1e-5, min_8bit_size=16384)

if resume_from and os.path.isfile(resume_from+'/optimizer.pt'):
    print('Loaded optimizer checkpoint')
    optimizer.load_state_dict(torch.load(resume_from+'/optimizer.pt'))

random.seed(42)
model.config.use_cache = False
trained_steps = 0

for epoch in range(1, epochs+1):
    total_loss = 0
    
    offset = int(epoch*CONTEXT/epochs)
    dataset_contexts = [train_tokens[i-CONTEXT:i] for i in range(CONTEXT+offset, len(train_tokens)-offset, CONTEXT)]
    random.shuffle(dataset_contexts)

    print(f'Epoch {epoch} starting. {len(dataset_contexts)} samples in dataset.')
    model.train()
    data_iterator = range(int(len(dataset_contexts)/batch_size))
    for i in (pbar := tqdm(data_iterator)):
        trained_steps += 1
        batch = torch.stack(dataset_contexts[i:i+batch_size])
        input_ids = batch.to(device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss /= gradient_accumulation_steps
        loss.backward()
        
        if trained_steps % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_average_loss = total_loss/(i+1)
        pbar.set_description(f"epoch avg loss {epoch_average_loss:.3f}")
        
        if trained_steps % 40 == 0:
            model.eval()
            eval_model(model, tokenizer)        
            model.train()
        if trained_steps % 200 == 0:
            print('\nSaving checkpoint')
            model.save_pretrained(f'checkpoints/gpt-sw3-finetune-{trained_steps}')
            torch.save(optimizer.state_dict(), f'checkpoints/gpt-sw3-finetune-{trained_steps}/optimizer.pt')

        if trained_steps == max_train_steps:
            break
            
    model.save_pretrained('checkpoints/gpt-sw3-finetune-'+str(epoch)+'-epochs')
    torch.save(optimizer.state_dict(), f'checkpoints/gpt-sw3-finetune-{epoch}-epochs/optimizer.pt')
            
    if trained_steps == max_train_steps:
        break

model.config.use_cache = True
model.eval()

model.save_pretrained('finetuned-models/gpt-sw3-finetune')

eval_model(model, tokenizer, generate=True)
