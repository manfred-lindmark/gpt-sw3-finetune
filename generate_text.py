import torch
import random
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from tqdm import tqdm

warnings.filterwarnings("ignore")
device = "cuda"

CONTEXT = 2048
USE_LORA = True

# All GPT-SW3 models can be found here: https://huggingface.co/AI-Sweden-Models
# the base model set below can be a Huggingface link or a local folder:
# base_model_name = "AI-Sweden-Models/gpt-sw3-6.7b-v2" # Huggingface link
base_model_name = "base-models/gpt-sw3-6.7b-v2-bf16"
lora_name = 'checkpoints/gpt-sw3-finetune-1-epochs'


prompt = '''\n***\nDu lyssnar på Creepypodden med mig, Jack Werner. I introt hörde vi en beskrivning av en varelse som många säger sig ha sett men endast som en formlös kontur i dimma och mörker. Dagens andra berättelse kommer också handla om den. Det är en riktigt kuslig och välskriven historia som kommer från reddits underforum nosleep. Har du själv någonsin gått hem sent en kväll när dimman hunnit lägga sig, och sedan upplevt hur du ser skepnader som rör sig just utanför ditt synfält? Det har i alla fall Elsa, och vi ska få höra hennes berättelse nu, uppläst av'''


# Initialize Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-6.7b-v2",
                                          truncation_side='left',
                                          padding_side="left",
                                          add_eos_token=False,
                                          add_bos_token=False,)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",)

# Save disk space by saving a local copy of the base model in half precision
# base_model.save_pretrained(base_model_name.split('/')[-1]+'-bf16', max_shard_size="100GB")

if USE_LORA:
    model = PeftModel.from_pretrained(base_model, lora_name)
    print('Using QLORA', lora_name)
else:
    model = base_model
    print('Using base model', base_model_name)

# These settings worked well for me, but may need adjustment.
# If the model is too repetitive try increasing temperature,
# or if it's output is too random try lowering it.
# A good range to try is from 0.45 to 0.70.
sampling_settings = {'temperature': 0.55,
                    'repetition_penalty': 1.12,
                    'top_k': 40,
                    'top_p': 0.95,}

seed = random.randint(1, 10000)
set_seed(seed)
print('Seed:', seed)

generation_size = 384

model.eval()
all_tokens = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"][0].to(device)
for i in tqdm(range(6)):
    model_input = all_tokens[generation_size-CONTEXT:]
    with torch.inference_mode():
        generated_tokens = model.generate(
            inputs=model_input.unsqueeze(0),
            max_new_tokens=generation_size,
            do_sample=True,
            **sampling_settings,
        )[0]

    new_tokens = generated_tokens[len(model_input):]
    all_tokens = torch.cat([all_tokens, new_tokens])

story = tokenizer.decode(
                all_tokens,
                skip_special_tokens=True,
            )
    
with open('generated_text/output.txt', 'a+', encoding='utf-8') as f:
    f.write(f'\n\n************************\n\n{lora_name if USE_LORA else base_model_name}\n{str(sampling_settings)}\n\n{story}')
