from datasets import load_dataset
import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

dataset = load_dataset("berkouille/Dolly_Golf", split="train")
dataset = dataset.train_test_split(test_size=0.3)
train_dataset = dataset['train']

def call_lora(r_temp,lora_alpha_temp,target_modules_temp):
    peft_config = LoraConfig(
        lora_alpha=lora_alpha_temp,
        lora_dropout=0.1,
        target_modules=target_modules_temp,
        r=r_temp,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return peft_config

def call_tranining_arguments(output_dir, epochs, lr_type):
    args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = epochs,
        per_device_train_batch_size = 6,
        warmup_steps = 0.03,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        lr_scheduler_type=lr_type,
        disable_tqdm=True
    )
    return args

def call_trainer(model, peft_config, max_seq_length, tokenizer, args, train_dataset):
    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        args=args,
        train_dataset=train_dataset,
        )
    return trainer

# Download Mistral 7B instruct model , load_in_4bit=True --> loading only 4-bit version
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map='auto', load_in_4bit=True, use_cache=False)
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
mistral_tokenizer.padding_side = "right"

#Prepare model
peft_config = call_lora(r_temp=64, lora_alpha_temp=16, target_modules_temp=["q_proj", "v_proj", "k_proj", "o_proj"])
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Training
training_args = call_tranining_arguments(output_dir = "./mistral_instruct_golf", epochs = 5, lr_type = "constant")
trainer = call_trainer(model, peft_config, max_seq_length=512, tokenizer= mistral_tokenizer, args=training_args, train_dataset=train_dataset)

# kick off the finetuning job
trainer.train()
trainer.save_model("mistral_instruct_golf")