import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from dotenv import load_dotenv
import boto3
import numpy as np
import argparse
import json


from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoFeatureExtractor
from transformers.models.clap.modeling_clap import ClapAudioModel, ClapTextModel
from transformers.models.clap.configuration_clap import ClapTextConfig, ClapAudioConfig, ClapConfig

import random
random.seed(42)
import torch

from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.training import TrainingArguments, Trainer, AnalysisTrainer
from gliclass.data_processing import DataCollatorWithPadding, GLiClassDataset
from gliclass.audio_data_processing import GLiClassAudioDataset, S3Manager



def compute_metrics(p):
    predictions, labels = p
    labels = labels.reshape(-1)
    if args.problem_type == 'single_label_classification':
        preds = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    elif args.problem_type == 'multi_label_classification':
        predictions = predictions.reshape(-1)
        preds = (predictions > 0.5).astype(int)
        labels = np.where(labels>0.5, 1, 0)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        accuracy = accuracy_score(labels, preds)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    else:
        raise NotImplementedError(f"{args.problem_type} is not implemented.")

def main(args):
    load_dotenv()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if args.model_name is not None:
        model = GLiClassModel.from_pretrained(args.model_name, focal_loss_alpha=args.focal_loss_alpha,
                                                                focal_loss_gamma=args.focal_loss_gamma)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
        encoder_config = AutoConfig.from_pretrained(args.encoder_model_name)
        audiocfg = AutoConfig.from_pretrained(args.audio_model_name)
        audio_feature_extractor = AutoFeatureExtractor.from_pretrained(args.audio_model_name)

        glicalss_config = GLiClassModelConfig(
            encoder_config=encoder_config,
            encoder_model=args.encoder_model_name,
            audio_model_name=args.audio_model_name,
            audio_model_config=audiocfg,
            class_token_index=len(tokenizer),
            text_token_index=len(tokenizer)+1,
            audio_token_index=len(tokenizer)+2, 
            pooling_strategy=args.pooler_type,
            scorer_type=args.scorer_type,
            use_lstm=args.use_lstm,
            focal_loss_alpha=args.focal_loss_alpha,
            focal_loss_gamma=args.focal_loss_gamma,
            contrastive_loss_coef=args.contrastive_loss_coef,
            normalize_features=args.normalize_features,
            extract_text_features=args.extract_text_features,
            architecture_type=args.architecture_type,
            prompt_first=args.prompt_first,
            squeeze_layers=args.squeeze_layers,
            shuffle_labels=args.shuffle_labels
        )

        model = GLiClassModel(glicalss_config, from_pretrained=True)

        if args.architecture_type in  {'uni-encoder', 'bi-encoder-fused', 'encoder-decoder', 'audio-encoder', 'audio-bi-encoder'}:
            new_words = ["<<LABEL>>", "<<SEP>>", "<<AUDIO>>"]
            tokenizer.add_tokens(new_words, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

    model.to(device)


    model.config.problem_type = args.problem_type

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

    s3manager = S3Manager(
        s3_client= s3_client,
        local_cache_dir= "./cache", #TODO move to args
        max_load_workers= 8,  #TODO move to args
        preload_size= 2000, #TODO move to args
        remaining_preloaded_threshold= 100, #0 #TODO move to args
    )

    train_dataset = GLiClassAudioDataset(
        dataset_path= args.data_path,
        s3manager= s3manager,
        tokenizer= tokenizer,
        audio_features_extractor= audio_feature_extractor,
        max_length= args.max_length,
        problem_type= args.problem_type,
        architecture_type= args.architecture_type,
        sampling_rate= args.sampling_rate,
        max_duration_s= args.max_duration_s,
        validate_json_file = True, #TODO move to args
        buffer_size = 8192 #TODO move to args
    )

    data_collator = DataCollatorWithPadding(device=device)

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    total_examples = train_dataset.get_num_examples()
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = total_examples // effective_batch_size
    max_steps = steps_per_epoch * args.num_epochs

    training_args = TrainingArguments(
        dataloader_pin_memory=False,
        output_dir=args.save_path,
        learning_rate=args.encoder_lr,
        weight_decay=args.encoder_weight_decay,
        audio_lr = args.audio_lr,
        audio_weight_decay = args.audio_weight_decay,
        others_lr=args.others_lr,
        others_weight_decay=args.others_weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_steps=max_steps,
        num_train_epochs=args.num_epochs,
        save_steps = args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers = args.num_workers,
        logging_steps=5,#100,
        use_cpu = False,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        accelerator_config={
            'dispatch_batches': False,
            'split_batches': False,
            'even_batches': True,    
        }
        )
    
    args_to_save = {
        "args": vars(args),
    }
    
    metrics_output_path = os.path.join(args.save_path, "args.json")
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    
    with open(metrics_output_path, "w") as f:
        json.dump(args_to_save, f, indent=4)
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    
    results_to_save = {
        "args": vars(args),
    }
    
    metrics_output_path = os.path.join(args.save_path, "training_results.json")
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    
    with open(metrics_output_path, "w") as f:
        json.dump(results_to_save, f, indent=4)
    
    print(f"Training metrics and arguments saved to {metrics_output_path}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default= None)
    parser.add_argument('--encoder_model_name', type=str, default = "microsoft/deberta-v3-base")
    parser.add_argument('--audio_model_name', type=str, default = "facebook/hubert-large-ls960-ft") # 
    parser.add_argument('--save_path', type=str, default = "./models/10M-gliclas-hu-audio-base")
    parser.add_argument('--data_path', type=str, default =  "./data/gliclass-audio-10M.jsonl")
    parser.add_argument('--problem_type', type=str, default='multi_label_classification')
    parser.add_argument('--pooler_type', type=str, default='first')
    parser.add_argument('--scorer_type', type=str, default='audio-token-dot')
    parser.add_argument('--architecture_type', type=str, default='audio-encoder')
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--extract_text_features', type=bool, default=False)
    parser.add_argument('--prompt_first', type=bool, default=True)
    parser.add_argument('--use_lstm', type=bool, default=False)
    parser.add_argument('--squeeze_layers', type=bool, default=False)
    parser.add_argument('--shuffle_labels', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=1) #££££££££££££££
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--audio_lr', type=float, default=1e-5)
    parser.add_argument('--others_lr', type=float, default=1e-5)
    parser.add_argument('--encoder_weight_decay', type=float, default=0.015)
    parser.add_argument('--audio_weight_decay', type=float, default=0.015)
    parser.add_argument('--others_weight_decay', type=float, default=0.015)
    parser.add_argument('--warmup_ratio', type=float, default=0.008) # approx 10k steps if data size ~1M
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--focal_loss_alpha', type=float, default=0.6)
    parser.add_argument('--focal_loss_gamma', type=float, default=2)
    parser.add_argument('--contrastive_loss_coef', type=float, default=0.)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--sampling_rate', type= int, default= 16000)
    parser.add_argument('--max_duration_s', type= int, default= 15, help="Max allowed duration of audio segment in seconds")
    parser.add_argument('--save_steps', type=int, default=5000)
    parser.add_argument('--save_total_limit', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--bf16', type=bool, default=True)
    args = parser.parse_args()

    main(args)
