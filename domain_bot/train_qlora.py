import argparse, os, json
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import Trainer

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def format_samples(rows, tokenizer, max_len=1024):
    prompts = []
    for r in rows:
        instr = r.get("instruction","")
        inp = r.get("input","")
        out = r.get("output","")
        text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        prompts.append(text)
    enc = tokenizer(prompts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    return enc, prompts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("[*] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora)

    print("[*] Loading and formatting dataset...")
    rows = load_jsonl(args.dataset_path)
    enc, _ = format_samples(rows, tokenizer)

    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, enc): self.enc = enc
        def __len__(self): return self.enc["input_ids"].shape[0]
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.enc.items()}

    ds = SimpleDS(enc)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_steps=200,
        fp16=False, bf16=True,
        optim="paged_adamw_32bit",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator
    )

    print("[*] Training...")
    trainer.train()
    print("[*] Saving LoRA adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
