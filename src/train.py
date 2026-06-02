import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict

try:
    from codecarbon import EmissionsTracker
    _has_codecarbon = True
except ImportError:
    _has_codecarbon = False

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

import src.domlm as model
import src.dataset as dataset
from src.data_collator import DataCollatorForDOMNodeMask


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     type=str, default=str(ROOT / 'data/swde_preprocessed'))
    parser.add_argument('--output_dir',   type=str, default=str(ROOT / 'results'))
    parser.add_argument('--domains',      type=str, default='university')
    parser.add_argument('--epochs',       type=int,   default=5)
    parser.add_argument('--batch_size',   type=int,   default=6)   # 6/GPU × 4 GPU = effective 24 (papier)
    parser.add_argument('--grad_accum',   type=int,   default=1)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--bf16',  action='store_true', help='BF16 mixed precision (Ampere+)')
    parser.add_argument('--tf32',  action='store_true', help='TF32 matmul (Ampere+)')
    parser.add_argument('--fp16',  action='store_true', help='FP16 mixed precision (Volta+)')
    parser.add_argument('--dataloader_workers', type=int, default=8)
    parser.add_argument('--save_steps',   type=int, default=500)
    parser.add_argument('--eval_steps',   type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    domains = args.domains.split(',')

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta = AutoModel.from_pretrained("roberta-base")

    roberta_config_dict = roberta.config.to_dict()
    roberta_config_dict["_name_or_path"] = "domlm"
    roberta_config_dict["architectures"] = ["DOMLMForMaskedLM"]
    domlm_config = model.DOMLMConfig.from_dict(roberta_config_dict)
    domlm = model.DOMLMForMaskedLM(domlm_config)

    state_dict = OrderedDict((f"domlm.{k}", v) for k, v in roberta.state_dict().items())
    domlm.load_state_dict(state_dict, strict=False)

    print(f"Loading datasets from {args.data_dir} — domains: {domains}")
    train_ds = dataset.SWDEDataset(args.data_dir, domain=domains, split="train")
    eval_ds  = dataset.SWDEDataset(args.data_dir, domain=domains, split="test")
    print(f"Train: {len(train_ds)} samples | Eval: {len(eval_ds)} samples")

    data_collator = DataCollatorForDOMNodeMask(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        tf32=args.tf32,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        report_to="none",
    )

    trainer = Trainer(
        model=domlm,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    tracker = None
    if _has_codecarbon:
        tracker = EmissionsTracker(
            project_name="domlm_pretrain",
            output_dir=args.output_dir,
            log_level="warning",
        )
        tracker.start()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if tracker is not None:
        emissions = tracker.stop()
        print(f"Carbon emissions: {emissions:.4f} kgCO2eq")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == '__main__':
    main()
