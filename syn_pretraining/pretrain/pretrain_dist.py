from transformers import GPT2Config
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from data import get_dataloader, GPT2Dataset, GPT2Collator, worker_init_fn, load_synthetic_dataset, SyntheticTokenizer, SyntheticDataset
import math
import csv
import time
import torch
import argparse
import os
from accelerate import Accelerator
import sys
from datasets import load_dataset


##########Huggingface multi GPU pretraining#########
@torch.no_grad()
def eval(args, dataloader, model, device, accelerator):
    model.eval()
    cumulative_loss = 0.
    total_rows = 0
    for step, batch in tqdm(enumerate(dataloader)):
        outputs = model(batch['input_ids'], attention_mask=batch["attention_mask"], labels=batch["input_ids"].detach().clone().long())  # fix here
        loss = outputs.loss
        accelerator.wait_for_everyone()
        gathered_loss = torch.mean(accelerator.gather(loss)).detach().cpu().item()
        cumulative_loss += gathered_loss
    return cumulative_loss / (step + 1)
    

def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    if "random" in os.path.basename(args.dataset):
        nvocab = int(os.path.basename(args.dataset).split("_")[3].split(".")[0])
    else:
        nvocab = int(os.path.basename(args.dataset).split("_")[4]) #hack
    args.nvocab = nvocab
    tokenizer = SyntheticTokenizer(nvocab)
    config = GPT2Config(
        vocab_size=nvocab,
        n_embd=args.embed_dim,
        n_layer=args.n_layer,
        n_head=args.n_head,
        activation_function=args.activation_function,
        n_positions=1024,
        resid_pdrop=args.dropout,
        use_cache=not args.grad_checkpoint,
        gradient_checkpointing=args.grad_checkpoint
    )

    model = GPT2LMHeadModel(config)
    if args.load_checkpoint:
        state_dict = torch.load(args.load_checkpoint)
        model.load_state_dict(state_dict)
        accelerator.accelerator.print(f"Loaded from {variant['load_checkpoint']}")
    train_data, valid_data, test_data = load_synthetic_dataset(args.dataset)

    with accelerator.main_process_first():
        valid_dataset = SyntheticDataset(valid_data, split="valid")
        test_dataset = SyntheticDataset(test_data, split="test")

    collator = GPT2Collator()

    batch_size = args.batch_size // 1024
    num_steps = args.num_steps
    num_warmup_steps = args.warmup_steps
    num_tokens = args.batch_size - args.batch_size // 1024

    test_data_loader = get_dataloader(
        test_dataset,
        collator,
        batch_size=batch_size,  
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        drop_last=False
    )

    valid_data_loader = get_dataloader(
        valid_dataset,
        collator,
        batch_size=batch_size,  
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        drop_last=False
    )

    if accelerator.is_main_process:
        with open(os.path.join(args.outdir, "progress.csv"), "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["current_train_time", "current_eval_time", "total_time", "steps", "train_loss", "train_ppl", "eval_ppl"])
    accelerator.wait_for_everyone()
    if not args.eval_only:
        with accelerator.main_process_first():
            train_dataset = SyntheticDataset(train_data, seed=args.seed, data_size=args.data_size)

        train_data_loader = get_dataloader(
            train_dataset,
            collator,
            batch_size=batch_size,
            worker_init_fn=worker_init_fn,
            num_workers=args.num_workers,
            drop_last = True
        )

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        (train_data_loader, valid_data_loader, test_data_loader, optimizer, model) = accelerator.prepare(train_data_loader, valid_data_loader, test_data_loader, optimizer, model)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_steps
        )

        scheduler = accelerator.prepare_scheduler(scheduler)

        total_start_time = time.time()
        cur_start_time = time.time()
        cumulative_loss = 0.
        cur_step = 0
        epoch = 0
        while True:
            for step, batch in tqdm(enumerate(train_data_loader), total=num_steps):
                if cur_step >= num_steps:
                    break
                model.train()
                labels = batch["input_ids"].detach().clone().long()
                outputs = model(
                    batch['input_ids'], 
                    attention_mask=batch["attention_mask"], 
                    labels=batch["input_ids"].detach().clone().long()
                    )
                optimizer.zero_grad()  
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.wait_for_everyone()
                gathered_loss = torch.mean(accelerator.gather(loss)).detach().cpu().item()
                cumulative_loss += gathered_loss
                optimizer.step()
                scheduler.step()
                accelerator.wait_for_everyone()
                if (cur_step + 1) % args.num_steps_per_save == 0:
                    torch.cuda.empty_cache()
                    train_ppl = torch.exp(torch.tensor(cumulative_loss) / args.num_steps_per_save).item()
                    cur_train_end_time = time.time()
                    if not args.no_eval:
                        valid_loss = eval(args, valid_data_loader, model, device, accelerator)
                        valid_ppl = math.exp(valid_loss)
                        cur_eval_end_time = time.time()
                        if accelerator.is_main_process:
                            accelerator.print(f'Step {cur_step + 1}/{num_steps}, epoch {epoch}, train loss = {cumulative_loss}, train ppl = {train_ppl}, valid ppl = {valid_ppl}')
    
                        cur_total_time = time.time()
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            if args.outdir is not None:
                                accelerator.unwrap_model(model).save_pretrained(f"{args.outdir}/model_{cur_step + 1}")
                            with open(os.path.join(args.outdir, "progress.csv"), "a") as f:
                                writer = csv.writer(f, delimiter="\t")
                                writer.writerow([cur_train_end_time - cur_start_time, cur_eval_end_time - cur_train_end_time, cur_total_time - total_start_time, cur_step + 1, cumulative_loss, train_ppl, valid_ppl])
                        accelerator.wait_for_everyone()
                        cur_start_time = time.time()
                        cumulative_loss = 0.
                        torch.cuda.empty_cache()
                    else:
                        if accelerator.is_main_process:
                            accelerator.print(f'Step {cur_step + 1}/{num_steps}, epoch {epoch}, train loss = {cumulative_loss}, train ppl = {train_ppl}')

                        cur_total_time = time.time()
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            if args.outdir is not None:
                                accelerator.unwrap_model(model).save_pretrained(f"{args.outdir}/model_{cur_step + 1}")
                            with open(os.path.join(args.outdir, "progress.csv"), "a") as f:
                                writer = csv.writer(f, delimiter="\t")
                                writer.writerow([cur_train_end_time - cur_start_time, -1, cur_total_time - total_start_time, cur_step + 1, cumulative_loss, train_ppl, -1])
                        accelerator.wait_for_everyone()
                        cur_start_time = time.time()
                        cumulative_loss = 0.
                        torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                cur_step +=1
            epoch += 1
            if cur_step >= num_steps:
                break
    else:
        (valid_data_loader, test_data_loader, model) = accelerator.prepare(valid_data_loader, test_data_loader, model)

    if not args.no_eval:
        test_start_time = time.time()
        test_loss = eval(args, test_data_loader, model, device, accelerator)
        test_ppl = math.exp(test_loss)
        test_end_time = time.time()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print(f'Training complete, final test ppl = {test_ppl}')
            with open(os.path.join(args.outdir, "progress.csv"), "a") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([0., test_end_time - test_start_time, test_end_time - total_start_time, "test", 0., 0., test_ppl])
            if args.outdir is not None:
                accelerator.unwrap_model(model).save_pretrained(f"{args.outdir}/model_final")
        return accelerator
    else:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print(f'Training complete')
            with open(os.path.join(args.outdir, "progress.csv"), "a") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([0., -1, -1, "test", 0., 0., -1])
            if args.outdir is not None:
                accelerator.unwrap_model(model).save_pretrained(f"{args.outdir}/model_final")
        return accelerator


def set_dt_args(args_to_parse=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str,
    )  
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument(
        "--model_type", type=str, default="dt"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=80000)
    parser.add_argument("--num_steps_per_save", type=int, default=10000)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--output_fname", type=str, default="progress.csv")
    parser.add_argument("--grad_checkpoint", action="store_true", default=False)
    parser.add_argument("--data_size", type=float, default=1.0)

    parser.add_argument("--no_eval", action="store_true", default=False)


    if args_to_parse is not None:
        args = parser.parse_args(args_to_parse)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    start_time = time.time()
    args = set_dt_args()
    data_dir = 'pretrain/checkpoints_ngram'
    exp_name_full = args.outdir
    args.outdir = os.path.join(data_dir, exp_name_full)
    os.makedirs(args.outdir, exist_ok=True)
    if args.load_checkpoint is not None:
        args.load_checkpoint = os.path.join(data_dir, args.load_checkpoint)
    accelerator = main(args)
    accelerator.print("Total time used: %.3f hours." % ((time.time() - start_time)/3600))

