from unsloth import FastModel
import argparse
import torch
import random
import numpy as np
import pandas as pd
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from unsloth.chat_templates import train_on_responses_only
import re
from sklearn.metrics import f1_score
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./saved_model')
    parser.add_argument('--pred_csv', type=str, default='predictions.csv')
    return parser.parse_args()


def main():
    print("added new prompt")
    args = parse_args()
    set_seed(args.seed)

    # Load model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3-27b-it",
        max_seq_length = 2048,
        load_in_4bit = False,
        load_in_8bit = False,
        full_finetuning = False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 8,
        lora_alpha = 8,
        lora_dropout = 0,
        bias = "none",
        random_state = args.seed,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )

    # Load and preprocess train dataset
    data_files = {"train": args.train_file}
    dataset = load_dataset("json", data_files=data_files)
    train_dataset = standardize_data_formats(
        dataset["train"]
    )

    def formatting_prompts_func(examples):
       convos = examples["messages"]
       texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
       return { "text" : texts, }
    
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps = 10,
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed=args.seed,
            report_to = "none",
            dataset_num_proc=2,

        ),
    )


    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
)
    trainer.train()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)



    # Inference on test set
    def call_llm(messages):

        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
        )
        outputs = model.generate(
            **tokenizer([text], return_tensors = "pt").to("cuda"),
            max_new_tokens = 64, # Increase for longer outputs!
            # Recommended Gemma-3 settings!
            temperature = 1.0, top_p = 0.95, top_k = 64,
        )
        s = tokenizer.batch_decode(outputs)
        return convert_unsloth_to_openai_format(s[0])[-1]['content']

    def classify_csv_to_dataframe(csv_path: str, system_prompt: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        predictions = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            content = str(row["Content"]).strip()

            if not content:
                predictions.append(None)
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]


            predicted_class = call_llm(messages)
            predictions.append(predicted_class)

        df["PredictedClass"] = predictions
        return df


    def assert_predictions_in_class(df: pd.DataFrame):
        valid_labels = set(df["Class"].dropna().unique())
        invalid_preds = df[~df["PredictedClass"].isin(valid_labels)]

        if not invalid_preds.empty:
            invalid_values = invalid_preds["PredictedClass"].unique()
            print(f"PredictedClass contains invalid label(s): {invalid_values.tolist()}")
            print(f"Valid labels are: {sorted(valid_labels)}")
        else:
            print("PredictedClass: invalid_preds is null.")



    def evaluate_macro_f1(df: pd.DataFrame) -> float:
        def merge_labels(label):
            # Merge 'Neutral' and 'Mixed' into 'Neutral_Mixed'
            if label == 'Neutral' or label == 'Mixed':
                return 'Neutral_Mixed'
            return label
        true_labels_merged = [merge_labels(x) for x in df["Class"].astype(str)]
        predicted_labels_merged = [merge_labels(x) for x in df["PredictedClass"].astype(str)]
        macro_f1 = f1_score(true_labels_merged, predicted_labels_merged, average='macro')

        return macro_f1

    def convert_unsloth_to_openai_format(text):
        # Remove <bos> if present
        text = text.replace("<bos>", "").strip()
        
        # Split by <start_of_turn>...<end_of_turn> blocks
        pattern = r"<start_of_turn>(.*?)<end_of_turn>"
        turns = re.findall(pattern, text, flags=re.DOTALL)

        messages = []

        for turn in turns:
            # First word is the role (user/model/system), rest is content
            lines = turn.strip().split("\n", 1)
            if len(lines) == 2:
                role_token, content = lines
            else:
                role_token, content = "unknown", lines[0]

            role_token = role_token.strip().lower()

            # Map role tokens to OpenAI role labels
            if role_token == "user":
                role = "user"
            elif role_token == "model":
                role = "assistant"
            elif role_token == "system":
                role = "system"
            else:
                role = "user"  # fallback

            messages.append({
                "role": role,
                "content": content.strip()
            })

        return messages

    print("Running inference on test set...")

    system_prompt = "Choose only one sentiment between: Positive, Negative, Neutral, or Mixed for each user input. Only return the classification and nothing else."
    df_with_predictions = classify_csv_to_dataframe(args.test_file, system_prompt)
    df_with_predictions.to_csv(args.pred_csv, index=False)
    print(f"Predictions saved to {args.pred_csv}")
    macro_f1 = evaluate_macro_f1(df_with_predictions)
    print("Macro F1 Score:", macro_f1)

    assert_predictions_in_class(df_with_predictions)
if __name__ == "__main__":
    main()
