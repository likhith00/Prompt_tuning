# Prompt_tuning
This repository is the implementation of prompt based parameter efficient finetuning technique called prompt tuning

# What is Prompt Tuning? 
Prompt tuning is a method for text classification tasks, primarily designed for T5 models, that reformulates downstream tasks as text generation problems. Instead of assigning a class label to a text sequence, tokens representing the class label are generated. A prompt, in the form of a series of tokens, is added to the input.In prompt tuning, the key concept is that prompt tokens have their own independent parameters, which are updated during training while keeping the pretrained model's parameters frozen. This approach allows for the optimization of the prompt tokens without changing the model parameters. Prompt tuning has been shown to yield comparable results to traditional training methods and maintains its performance advantage as the model size increases. This technique focuses on optimizing the input prompt, offering a more adaptable and efficient alternative to traditional fine-tuning methods.

## Model 

bigscience/bloomz-560m model: can follow human instructions in multiple languages without specific training for each language (zero-shot). These models are fine-tuned from pretrained multilingual language models, such as BLOOM and mT5, using a cross-lingual task mixture called xP3. The fine-tuned models demonstrate cross-lingual generalization to unseen tasks and languages, making them versatile and adaptable for various NLP applications in multiple languages.

## Dataset

twitter_complaints used in this prompt tuning task. It subset of the RAFT dataset. The twitter_complaints subset contains tweets labeled as complaint and no complaint

## libraries used

- peft: for model pruning and quantization
- transformers: transformers: For utilizing and fine-tuning the model.
- datasets: For handling and processing the data.
- numpy: For numerical computations.
- torch: For building and training neural networks.

## Hyper parameters

- learning rate = 3e-2
- num_epochs = 50
- batch_size = 8

# Usage

`
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


model = AutoPeftModelForCausalLM.from_pretrained("likhith231/bloomz-560m_PROMPT_TUNING_CAUSAL_LM_50").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

inputs = tokenizer(
    "Tweet text : Couples wallpaper, so cute. :) #BrothersAtHome. Label : ",
    return_tensors="pt",
)
import torch
with torch.no_grad():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
    )
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
`


