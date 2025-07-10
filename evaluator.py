from transformers import AutoTokenizer, AutoConfig
from gliclass import GLiClassModelConfig, GLiClassModel
from gliclass.pipeline import BaseZeroShotClassificationPipeline, AudioEncoderZeroShotClassificationPipeline, ZeroShotClassificationPipeline
import torch, torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from gliclass.data_processing import GLiClassDataset
from dataclasses import dataclass
import json
import os
import random
import numpy as np
from tqdm import tqdm
import warnings
from pprint import pprint
random.seed(42)


@dataclass
class EvaluationObject:
    name: str
    model: GLiClassModel
    tokenizer: AutoTokenizer
    audio_feature_extractor: Wav2Vec2FeatureExtractor
    
    def get_name(self):
        return self.name
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_features_extractor(self):
        return self.audio_feature_extractor
    

class Evaluaor:
    def __init__(
        self,
        dataset_name: str,
        loader_fn: callable,
        models_names: list,
        max_length: int = 1024,
        eval_subset_size: int = 5000 
    ) -> None:
        self.dataset = self.load_dataset(dataset_name, loader_fn, eval_subset_size)
        self.all_labels = self.collect_all_labels()
        self.evaluation_objects = self.load_models(models_names)
        self.max_length = max_length
    
    def load_dataset(self, dataset_name: str, loader_fn, eval_subset_size):
        dataset = loader_fn(dataset_name)
        return dataset[:eval_subset_size]

    def collect_all_labels(self):
        all_labels = set()
        for row in self.dataset:
            if 'all_labels' in row:
                all_labels.update(row['all_labels'])
            else:
                warnings.warn(f"missing collumn all_labels for {row['id']} row", UserWarning)
        print("Tottal unique labels: ", len(all_labels))
        return list(all_labels)

    def load_models(self, models_names):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        evaluation_objects = []
        for model_name in models_names:
            model = GLiClassModel.from_pretrained(model_name).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model.config.audio_model_config._name_or_path,
            )

            evaluation_objects.append(
                EvaluationObject(
                    name= model_name,
                    model= model,
                    tokenizer= tokenizer,
                    audio_feature_extractor=audio_feature_extractor
                )
            )

        return evaluation_objects

    def prepare_input(self, labels, prompt_first):
        input_text = []
        for label in labels:
            label_tag = f"<<LABEL>>{label.lower()}"
            input_text.append(label_tag)
        input_text.append("<<SEP>>")
        if prompt_first:
            input_text = "".join(input_text) + "<<AUDIO>>"
        else:
            input_text = "<<AUDIO>>" + "".join(input_text)
        return input_text

    def calculate_results(self, logits, true_labels, all_labels_batch, threshold: float = -1):
        probabilities = torch.nn.functional.sigmoid(logits)

        predicted_classes = []
        for i, (sample_probs, sample_labels) in enumerate(zip(probabilities, all_labels_batch)):
            if threshold > 0:
                predictions = (sample_probs > threshold).cpu().numpy()
                classes = [sample_labels[j] for j, pred in enumerate(predictions) if pred]
            else:
                best_idx = torch.argmax(sample_probs).item()
                classes = [sample_labels[best_idx]]
            
            predicted_classes.append(classes)

        tp = {label: 0 for label in self.all_labels}
        fp = {label: 0 for label in self.all_labels}
        tn = {label: 0 for label in self.all_labels}
        fn = {label: 0 for label in self.all_labels}
        for pred_labels, true_labels in zip(predicted_classes, true_labels):
            for label in self.all_labels:
                pred_is_label = (label in pred_labels) 
                true_is_label = (label in true_labels)
                
                if pred_is_label and true_is_label:
                    tp[label] += 1 
                elif pred_is_label and not true_is_label:
                    fp[label] += 1  
                elif not pred_is_label and not true_is_label:
                    tn[label] += 1  
                else:  
                    fn[label] += 1 
        return tp, fp, tn , fn
    
    def calculate_metrics(self, tp, fp, tn, fn):
        precision = {}
        recall = {}
        f1 = {}
        
        for label in self.all_labels:
            # Precision: TP / (TP + FP)
            precision[label] = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
            
            # Recall: TP / (TP + FN)
            recall[label] = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
            
            # F1-score: 2 * (precision * recall) / (precision + recall)
            f1[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) > 0 else 0

        macro_precision = sum(precision.values()) / len(precision)
        macro_recall = sum(recall.values()) / len(recall)
        macro_f1 = sum(f1.values()) / len(f1)

        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_tn = sum(tn.values())
        total_fn = sum(fn.values())

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1
        }

    def get_clean_results(self, full_results):
        clean_results = {}
        for model_name, results in full_results.items():
            clean_results[model_name] = {
                "macro_precision": results["macro_precision"],
                "macro_recall": results["macro_recall"],
                "macro_f1": results["macro_f1"],
                "micro_precision": results["micro_precision"],
                "micro_recall": results["micro_recall"],
                "micro_f1": results["micro_f1"]
            }
        return clean_results

    def evaluate(self, batch_size: int = 8, return_clean = True):
        models_results = {}
        for object_ in self.evaluation_objects:
            name = object_.get_name()
            model = object_.get_model()
            tokenizer = object_.get_tokenizer()
            audio_feature_extractor = object_.get_features_extractor()
            prompt_first = model.config.prompt_first

            print("Evaluating model: ", name)
            model.eval()
            models_results[name]= {
                "tp" : {label: 0 for label in self.all_labels},
                "fp" : {label: 0 for label in self.all_labels},
                "tn" : {label: 0 for label in self.all_labels},
                "fn" : {label: 0 for label in self.all_labels}
            }

            for i in tqdm(range(0, len(self.dataset), batch_size), desc= "Evaluating"):
                batch_rows = self.dataset[i:i+batch_size]

                audio_raw_list = []
                prompts = []
                all_labels_batch = [] 
                true_labels_batch = []
                for row in batch_rows:
                    prompts.append(self.prepare_input(row["all_labels"], prompt_first))
                    all_labels_batch.append(row["all_labels"]) 
                    true_labels_batch.append(row["true_labels"]) 

                    audio_raw = torch.load(row['audio_path']).float()
                    sample_rate = row["sample_rate"]
                    
                    if sample_rate != 16000:
                        audio_raw = torchaudio.transforms.Resample(
                            orig_freq=sample_rate, new_freq=16000
                        )(audio_raw)
                                        
                    audio_raw_list.append(audio_raw.numpy())

                audio_inputs = audio_feature_extractor(
                    audio_raw_list,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=16000 * 15
                )

                tokenized_inputs = tokenizer(
                    prompts, 
                    truncation=True, 
                    max_length=self.max_length, 
                    padding="longest", 
                    return_tensors="pt"
                ).to(self.device)

                labels_mask = torch.ones(len(batch_rows), len(all_labels_batch)).to(self.device)
                tokenized_inputs["labels_mask"] = labels_mask
                tokenized_inputs["input_audio_features"] = audio_inputs["input_values"].to(self.device)
                tokenized_inputs["audio_attention_mask"] = audio_inputs["attention_mask"].to(self.device)

                with torch.no_grad():
                    logits = model(**tokenized_inputs).logits
                batch_tp, batch_fp, batch_tn, batch_fn = self.calculate_results(logits, true_labels_batch, all_labels_batch)
                for label in self.all_labels:
                    models_results[name]["tp"][label] += batch_tp[label]
                    models_results[name]["fp"][label] += batch_fp[label]
                    models_results[name]["tn"][label] += batch_tn[label]
                    models_results[name]["fn"][label] += batch_fn[label]

            metrics = self.calculate_metrics(
                models_results[name]["tp"],
                models_results[name]["fp"],
                models_results[name]["tn"],
                models_results[name]["fn"]
            )
            models_results[name].update(metrics)
            if return_clean:
                pprint(self.get_clean_results(models_results))

            if return_clean:
                models_results = self.get_clean_results(models_results)
        return models_results

def load_emotions_dataset(dataset_name="Hemg/Emotion-audio-Dataset", metadata_path: str = "./datasets/metadata.json"):
    # print("Loading dataset: ", dataset_name)
    # with open(metadata_path, "r", encoding='utf-8') as f:
        # metadata = json.load(f)
    # dataset_path = os.path.join(os.path.dirname(metadata_path), metadata["dataset_file_name"])
    file_path = "./datasets/eval_emotions.json"
    print ("LOADING FILE: ", file_path)
    with open(file_path, "r", encoding='utf-8') as f:
        dataset = json.load(f)
    random.shuffle(dataset)
    return dataset

def load_synthetic_vocal_bursts_splittrain(dataset_name="synthetic_vocal_bursts_splittrain", data_path: str = "./datasets/synthetic_vocal_bursts_splittrain/laion-synthetic_vocal_bursts_splittrain.json"):
    print("Loading dataset: ", dataset_name)
    with open(data_path, "r", encoding='utf-8') as f:
        dataset = json.load(f)
    random.shuffle(dataset)
    return dataset

def load_LAION_Audio_300M_splittrain(dataset_name="LAION-Audio-300M_splittrain", data_path: str = "./datasets/LAION-Audio-300M_splittrain/laion-LAION-Audio-300M_splittrain.json"):
    print("Loading dataset: ", dataset_name)
    with open(data_path, "r", encoding='utf-8') as f:
        dataset = json.load(f)
    random.shuffle(dataset)
    return dataset

######### RUNNERS FUNC ################

def eval_emotions(eval_size):
    DATASET_NAME = "Hemg/Emotion-audio-Dataset"
    evaluator = Evaluaor(
        dataset_name= DATASET_NAME,
        loader_fn = load_emotions_dataset,
        models_names= [
            # "werent4/1M-gliclas-hu-audio-base-chp45k",
            # "werent4/1M-gliclas-hu-audio-base-chp60k",
            # "werent4/1M-gliclas-hu-audio-base-chp85k",
            # "werent4/1M-gliclas-hu-audio-base-chp100k",
            # "werent4/1M-gliclas-hu-audio-base-chp115k",
            # "werent4/1M-gliclas-hu-audio-base-v1 "
            # "models/scratch-gliclas-hu-audio-emo/checkpoint-7200",
            # "models/1M-gliclas-hu-audio-base-v1-emo/checkpoint-1440"
            "models/1M-gliclas-hu-audio-base-v1-emo-ep-5/checkpoint-7200"
            # FEW SHOT MODELS
            # "models/gliclas-hu-audio-base-emo-2/checkpoint-16",
            # "werent4/gliclas-hu-audio-base-emo-4", #"models/gliclas-hu-audio-base-emo-4/checkpoint-24",
            # "models/gliclas-hu-audio-base-emo-6-ep-8-wr-03/checkpoint-40",
            # "models/gliclas-hu-audio-base-emo-8-wr-03/checkpoint-28",
            # "models/gliclas-hu-audio-base-emo-16-ep-4-wr-03/checkpoint-56",
            # "models/gliclas-hu-audio-base-emo-16-wr-03/checkpoint-28"
            # "models/gliclas-hu-audio-base-v1-emo-2/checkpoint-32",
            # "models/gliclas-hu-audio-base-v1-emo-4/checkpoint-48",
            # "models/gliclas-hu-audio-base-v1-emo-6/checkpoint-48",
            # "models/gliclas-hu-audio-base-v1-emo-8/checkpoint-28",
            # "models/gliclas-hu-audio-base-v1-emo-8-ep-8/checkpoint-56",
            # "models/gliclas-hu-audio-base-v1-emo-16/checkpoint-28",
            # "models/gliclas-hu-audio-base-v1-emo-16-ep-4/checkpoint-56",
            # "models/gliclas-hu-audio-base-v1-emo-16-ep-8/checkpoint-112",
        ],
        eval_subset_size= eval_size,
        max_length= 1024
    )
    results = evaluator.evaluate(batch_size= 12)
    print("Results for: ", DATASET_NAME)
    pprint(results)

def eval_synthetic_vocal_bursts_splittrain(eval_size):
    DATASET_NAME = "synthetic_vocal_bursts_splittrain"
    evaluator = Evaluaor(
        dataset_name= DATASET_NAME,
        loader_fn = load_synthetic_vocal_bursts_splittrain,
        models_names= [
            # "werent4/1M-gliclas-hu-audio-base-chp45k",
            # "werent4/1M-gliclas-hu-audio-base-chp60k",
            # "werent4/1M-gliclas-hu-audio-base-chp85k",
            # "werent4/1M-gliclas-hu-audio-base-chp100k",
            # "werent4/1M-gliclas-hu-audio-base-chp115k",
            "/mnt/storage-werent4-2tb/models/1M-gliclas-hu-audio-base/checkpoint-135480"
        ],
        eval_subset_size= eval_size,
        max_length= 1900
    )
    results = evaluator.evaluate(batch_size= 1)
    print("Results for: ", DATASET_NAME)
    pprint(results)

def eval_LAION_Audio_300M_splittrain(eval_size):
    DATASET_NAME = "LAION-Audio-300M_splittrain"
    evaluator = Evaluaor(
        dataset_name= DATASET_NAME,
        loader_fn = load_LAION_Audio_300M_splittrain,
        models_names= [
            # "werent4/1M-gliclas-hu-audio-base-chp45k",
            # "werent4/1M-gliclas-hu-audio-base-chp60k",
            # "werent4/1M-gliclas-hu-audio-base-chp85k",
            # "werent4/1M-gliclas-hu-audio-base-chp100k",
            # "werent4/1M-gliclas-hu-audio-base-chp115k",
            "/mnt/storage-werent4-2tb/models/1M-gliclas-hu-audio-base/checkpoint-135480"
        ],
        eval_subset_size= eval_size,
        max_length= 1280
    )
    results = evaluator.evaluate(batch_size= 1)
    print("Results for: ", DATASET_NAME)
    pprint(results)

def main():
    EVAL_SIZE = 5000
    # eval_LAION_Audio_300M_splittrain(EVAL_SIZE)
    # eval_synthetic_vocal_bursts_splittrain(EVAL_SIZE)
    eval_emotions(EVAL_SIZE)    


if __name__ == "__main__":
    main()