# Fine-tune GPT-SW3 using quantized LORA

This repository contains minimal code for demonstrating fine-tuning of the GPT-SW3 language models using the QLORA method. The code is written for use with a Nvidia GPU and tested on Windows. The training setup should work with any of the model sizes. With 24 GB of VRAM you can train the 6.7B model with a batch size of 2. To train with less memory you can reduce batch size, enable gradient checkpointing, reduce LORA rank and layers, or simply swap to a smaller base GPT model.

## Setting up Python requirements

Anaconda is a recommended way of installing the required Python packages.
The command below will create an environment with most requirements.

```bash
conda create --name gpt-sw3-finetune python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 transformers peft sentencepiece huggingface_hub -c pytorch -c nvidia -c huggingface -c conda-forge
```

Additionally you need "bitsandbytes" which does not have official Windows support. There are however precompiled versions available. Make sure your Anaconda environment is activated and run the following to install bitsandbytes on Windows. This is the latest version as of January 2024. On linux it can be installed with pip.

```bash
conda activate gpt-sw3-finetune
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

## Dataset (text corpus) for training

This example is written for smaller datasets in a single .txt file.
You can get a ~12 MB sample Swedish dataset containing horror stories from the "creepypodden" podcast by running get_dataset.py. This requires a few more python packages.

```bash
conda install anaconda::beautifulsoup4 anaconda::html5lib
python get_dataset.py
```

## Inference: just generating text

The file "generate_text.py" is meant for testing inference on the finetuned model. Edit the file to set a text generation prompt, generation length or sampling settings. You can compare with the base by setting USE_LORA to True/False. The generated text is saved to "generated_text/output.txt", and each run will be appended to the same file.

## Training

Run "train_gpt-sw3_qlora.py" to train a QLORA adapter model to a base GPT model. All settings can be changed by editing the file.

## Troubleshooting Python environment

New versions of the required Python packages are frequently released so if there are any problems you should follow the installation guides for each of them to check their compatibility.

## Troubleshooting training

Learning rate is a good first step to try increasing/decreasing.

## Output demo

### Prompt:

> Du lyssnar på Creepypodden med mig, Jack Werner. I introt hörde vi en beskrivning av en varelse som många säger sig ha sett men endast som en formlös kontur i dimma och mörker. Dagens andra berättelse kommer också handla om den. Det är en riktigt kuslig och välskriven historia som kommer från reddits underforum nosleep. Har du själv någonsin gått hem sent en kväll när dimman hunnit lägga sig, och sedan upplevt hur du ser skepnader som rör sig just utanför ditt synfält? Det har i alla fall Elsa, och vi ska få höra hennes berättelse nu, uppläst av

### Base model (GPT-SW3-6.7B-V2)

<blockquote>henne själv! Varsågoda:

Tack för att du gjorde denna pod till verklighet 🙂

Razmig:
Fantastisk podd! 😁

Kariann:
Helvete vad bra avsnitt! Nu måste jag lyssna igenom de gamla igen bara för det..

Masoumeh:
Jag vill tacka er för ett mycket spännande och underhållande program. Jag har lyssnat på samtliga avsnitt och ni gör ett fantastiskt jobb!

Jensine:
Grymt avsnitt! Älskar verkligen eran podcast och era historier. Ni verkar vara väldigt sköna grabbar! Keep up the good work!

Welday:
Härligt att ni gillar det! Nästa vecka blir det ännu mer läskigt! Vi spelar in i morgon förmiddag så det borde komma ut nästa måndag eller tisdag. Hoppas ni inte har något emot lite väntan!</blockquote>

### Same model finetuned with QLORA adapter

<blockquote> Rakel Josephson.<br>
Jag vet inte varför jag skriver detta. Jag vet att det finns de som tycker det här är sjukt, men jag kan inte hjälpa det. Jag vill bara berätta vad jag såg.<br>
Igår gick jag hem efter jobbet vid halv nio-tiden. När jag kom fram till huset märkte jag att ytterdörren stod öppen och att min bil var borta. Jag antog att någon hade tagit in bilen eftersom den inte var parkerad där den brukar stå, så jag gick in för att leta efter mina nycklar.<br>
När jag letade igenom jackan hittade jag dem inte och tänkte att jag kanske glömt dem någonstans, så jag tittade runt lite i huset utan att hitta dem. Sedan gick jag ut igen och började gå mot bilen. Då upptäckte jag något konstigt i ögonvrån. En mörk skugga som rörde sig långsamt bakom mig, åtföljd av ett svagt ljussken. Jag vände mig instinktivt om och fick se att skuggan befann sig precis bakom mig, ungefär fem meter bort. Ljuset verkade komma från ett litet fönster eller en dörr längre ner längs vägen. Jag försökte vrida huvudet åt sidan för att se vem det var, men kunde inte röra mig ur fläcken - det kändes som om någon höll fast mitt huvud. Jag ropade "Hallå?" utan resultat. Så fort jag öppnade munnen blev ljudet dämpat och försvann helt.</blockquote>
