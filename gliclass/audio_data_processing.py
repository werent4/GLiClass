import random
import torch
import threading
from torchaudio.transforms import Resample
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .augments import augment_audio
import os
from urllib.parse import urlparse
from pathlib import Path
import time
import warnings
from enum import Enum

class DownloadStatus(Enum):
    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"

def parse_s3_path(s3_path):
    if not s3_path.startswith('s3://'):
        return None, None
    
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key

def get_local_path(s3_path, local_cache_dir):
    bucket, key = parse_s3_path(s3_path)
    if bucket is None:
        return s3_path

    key_parts = key.split("/")
    subset_dir = key_parts[-2]
    filename = key_parts[-1]
    local_path = os.path.join(local_cache_dir, subset_dir, filename)
    return local_path

class GLiClassAudioDataset(IterableDataset):
    def __init__(
            self,
            examples,
            s3_client,
            tokenizer,
            max_length=512, 
            problem_type='multi_label_classification', 
            architecture_type = 'audio-encoder',
            prompt_first=False,
            get_negatives = False,
            max_labels = 150,
            audio_features_extractor=None,
            sampling_rate = 16000,
            max_duration_s = 15, # seconds
            shuffle_labels = True,
            local_cache_dir='../datasets/cache',
            max_load_workers=4,
            preload_size = 20
        ):
        if architecture_type != 'audio-encoder':
            raise ValueError("This class was specifecly created for 'audio-encoder' arch")
        
        self.s3_client = s3_client
        self.tokenizer = tokenizer
        self.audio_features_extractor = audio_features_extractor
        self.max_length = max_length
        self._data = examples
        self.problem_type = problem_type
        self.prompt_first = prompt_first
        self.dataset_labels = self.collect_dataset_labels()
        self.get_negatives = get_negatives
        self.max_labels = max_labels
        self.shuffle_labels = shuffle_labels
        print('Total labels: ', len(self.dataset_labels))

        if self.audio_features_extractor is not None:
            if sampling_rate is None or max_duration_s is None:
                raise ValueError(
                    "When using audio_features_extractor you must specify "
                    "sampling_rate и max_duration_s"
                )
            self.sampling_rate = sampling_rate
            self.max_duration_s = max_duration_s
            self.max_duration_samples = self.sampling_rate * self.max_duration_s
            print(f"Audio parameters: sampling_rate={self.sampling_rate}, "
                  f"max_duration={self.max_duration_s}s, "
                  f"max_samples={self.max_duration_samples}")
            
        self.need_preload = True
        self.preload_size = preload_size
        self.local_cache_dir = local_cache_dir
        self.max_load_workers = max_load_workers

        self.download_executor = ThreadPoolExecutor(max_workers=self.max_load_workers)
        self.loading_in_process = True
        self.file_status = {}  # local_path -> DownloadStatus; needs Lock
        self.download_futures = {}  # local_path -> Future object; needs Lock
        self.download_lock = threading.Lock()
        self.processed_files = set() # stores files which were already processed in main loop; needs Lock

        self.download_stats = { # needs Lock
            'total_requested': 0,
            'completed': 0,
            'failed': 0,
            'cached': 0
        }

    def collect_dataset_labels(self):
        dataset_labels = set()
        for example in self._data:
            dataset_labels.update(set(example['all_labels']))
        return dataset_labels
    
    def prepare_labels(self, example, label2idx, problem_type):
        if problem_type == 'single_label_classification':
            labels = label2idx[example['true_labels'][0]]
        elif problem_type == 'multi_label_classification':
            if isinstance(example['true_labels'], dict):
                labels = [example['true_labels'][label] if label in example['true_labels'] else 0. for label in example['all_labels']]
            else:
                labels = [1. if label in example['true_labels'] else 0. for label in example['all_labels']]
        else:
            raise NotImplementedError(f"{problem_type} is not implemented.")
        return torch.tensor(labels)

    def prepare_prompt(self, example):
        prompt_texts = []
        for label in example['all_labels']:
            label_tag = f"<<LABEL>>{str(label)}"
            prompt_texts.append(label_tag)
        prompt_texts.append('<<SEP>>')
        return prompt_texts
    
    def prepare_audio(self, audio_array, audio_sr):
        if isinstance(audio_array, np.ndarray):
            audio_array = torch.from_numpy(audio_array).float()
        elif isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.float()
        else:
            audio_array = torch.tensor(audio_array, dtype=torch.float32)

        # if random.random() > 0.3:
        #     audio_array = augment_audio(audio_array, audio_sr)

        if audio_sr != self.sampling_rate:
            audio_array = Resample(audio_sr, new_freq= self.sampling_rate)(audio_array)

        audio_inputs = self.audio_features_extractor(
            audio_array, 
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="longest",
            truncation=True, 
            max_length=self.max_duration_samples  
        )
        return audio_inputs["input_values"], audio_inputs["attention_mask"] 
    
    def tokenize(self, texts):
        tokenized_inputs = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding="longest", return_tensors="pt")
        return tokenized_inputs
    
    def tokenize_and_prepare_labels_for_audioencoder(self, example):
        if self.shuffle_labels:
            random.shuffle(example['all_labels'])
        input_text = self.prepare_prompt(example)
        input_text.append('<<AUDIO>>')
        input_text = ''.join(input_text)
        label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}
        

        tokenized_inputs = self.tokenize(input_text)
        tokenized_inputs['labels'] = self.prepare_labels(example, label2idx, self.problem_type)
        # tokenized_inputs['labels_text'] =  example['all_labels']

        audio_data = torch.load(example['audio_path'], weights_only=False)
        audio_sr = example["sample_rate"]
        tokenized_inputs["input_audio_features"], tokenized_inputs["audio_attention_mask"]  = self.prepare_audio(audio_data, audio_sr)
        return tokenized_inputs

    def get_file_status(self, local_path):
        with self.download_lock:
            return self.file_status.get(local_path, DownloadStatus.NOT_STARTED)

    def set_file_status(self, local_path, status):
        with self.download_lock:
            self.file_status[local_path] = status    

    def mark_file_as_processed(self, local_path):
        with self.download_lock:
            self.processed_files.add(local_path)

    def download_file(self, s3_path, local_path, is_last= False):
        bucket, key = parse_s3_path(s3_path)
        if bucket is None:
            return False
        try:
            self.set_file_status(local_path, DownloadStatus.DOWNLOADING)

            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=local_path
            )

            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                self.set_file_status(local_path, DownloadStatus.COMPLETED)
                with self.download_lock:
                    self.download_stats['completed'] += 1
                return True
            else:
                raise Exception("File not created or empty")
            
        except Exception as e:
            print(f"Failed to load {s3_path}: {e}")
            self.set_file_status(local_path, DownloadStatus.FAILED)
            with self.download_lock:
                self.download_stats['failed'] += 1
            return False
        
        finally:
            with self.download_lock:
                self.download_futures.pop(local_path, None)
            if is_last:
                self.loading_in_process = False
    
    def load_next(self, data_to_iterate, current_counter):
        if not self.need_preload:
            return    
        
        start_idx = current_counter + 1
        end_idx = min(start_idx + self.preload_size, len(data_to_iterate))

        if start_idx >= len(data_to_iterate):
            return
        
        submitted_count = 0
        for i in range(start_idx, end_idx):
            s3_path = data_to_iterate[i]['audio_path']
            local_path = get_local_path(s3_path, self.local_cache_dir)

            if os.path.exists(local_path):
                self.set_file_status(local_path, DownloadStatus.COMPLETED)
                with self.download_lock:
                    self.download_stats['cached'] += 1
                continue
            
            status = self.get_file_status(local_path)
            if status in [DownloadStatus.DOWNLOADING, DownloadStatus.COMPLETED]:
                continue
            
            is_last_in_batch = (i == end_idx - 1)
            with self.download_lock:
                self.download_stats['total_requested'] += 1
                future = self.download_executor.submit(self.download_file, s3_path, local_path, is_last_in_batch)
                self.download_futures[local_path] = future
                submitted_count += 1

        if submitted_count > 0:
            self.loading_in_process = True
            print(f"Started {submitted_count} new loads")

    def ensure_loaded(self, s3_path, timeout = 10) -> bool|str:
        local_path = get_local_path(s3_path, self.local_cache_dir)
        status = self.get_file_status(local_path)

        # skip iteration if failed
        if status == DownloadStatus.FAILED:
            return False
        
        # start loading if NOT starteed
        if status == DownloadStatus.NOT_STARTED:
            print(f"Emergency download: {s3_path}")
            with self.download_lock:
                self.download_stats['total_requested'] += 1
                future = self.download_executor.submit(self.download_file, s3_path, local_path)
                self.download_futures[local_path] = future
            status = DownloadStatus.DOWNLOADING
        
        # wait for `timeout` seconds for download
        if status == DownloadStatus.DOWNLOADING:
            print(f"Waiting for download: {s3_path}")

            start_time = time.time()
            last_log_time = start_time

            while time.time() - start_time < timeout:
                current_status = self.get_file_status(local_path)

                if current_status == DownloadStatus.COMPLETED and os.path.exists(local_path):
                    return local_path
                elif current_status == DownloadStatus.FAILED:
                    return False
                
                if time.time() - last_log_time >= 5:
                    elapsed = time.time() - start_time
                    print(f"Still waiting for file: {s3_path}; {elapsed:.0f}/{timeout}s: {local_path}")
                    last_log_time = time.time()
                time.sleep(0.5)

            print(f"Download timeout: {s3_path}")
            return False
        
        # if exists just return local path
        if os.path.exists(local_path) and status == DownloadStatus.COMPLETED:
            return local_path
        
        return False

    def _cleanup_batch(self, preloaded_data_start, preloaded_data_end, data_to_iterate):
        files_to_remove = []
        for i in range(preloaded_data_start, min(preloaded_data_end, len(data_to_iterate))):
            s3_path = data_to_iterate[i]['audio_path']
            local_path = get_local_path(s3_path, self.local_cache_dir)
            with self.download_lock:
                if (local_path in self.processed_files and 
                    local_path not in self.download_futures):
                    files_to_remove.append(local_path)

        removed_count = 0
        for local_path in files_to_remove:
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
                    removed_count += 1
            except Exception as e:
                print(f"Cleanup error: {e}")
            
            with self.download_lock:
                self.file_status.pop(local_path, None)
                self.processed_files.discard(local_path)
        print(f"Cleanup: removed {removed_count}/{len(files_to_remove)} files")


    def cleanup_processed_files(self, counter, data_to_iterate):
        preloaded_data_start = ((counter + 1) // self.preload_size - 1) * self.preload_size
        preloaded_data_end = preloaded_data_start + self.preload_size
        
        # we call additional thread here, this thread will make cleanup for us, to not stop main loop execution
        cleanup_thread = threading.Thread(
            target=self._cleanup_batch,
            args=(preloaded_data_start, preloaded_data_end, data_to_iterate),
            daemon=True
        )
        cleanup_thread.start()


    def __iter__(self):
        remaining_preloaded_threshold = 5
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            worker_id = "main"
            data_to_iterate = self._data
        else:
            per_worker = int(np.ceil(len(self._data) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self._data))
            print(f"worker_id: {worker_id}, [{iter_start}: {iter_end}]")
            data_to_iterate = self._data[iter_start:iter_end]

        self.load_next(data_to_iterate, -1)
        last_preloaded_index = min(self.preload_size - 1, len(data_to_iterate) - 1)
        print(f"Initial preload up to index: {last_preloaded_index}")

        for counter, example in enumerate(data_to_iterate):
            s3_path = example['audio_path']
            if not self.ensure_loaded(s3_path):
                warnings.warn(f"Skipping example {counter} as failed to load its audio file", UserWarning)
                continue
            local_path = get_local_path(s3_path, self.local_cache_dir)
            example['audio_path'] = local_path

            remaining_preloaded = last_preloaded_index - counter
            if remaining_preloaded <= remaining_preloaded_threshold and not self.loading_in_process:
                print(f"Triggering preload: {remaining_preloaded} files remaining (counter: {counter}, last_preloaded: {last_preloaded_index})")
                self.need_preload = True

            if self.need_preload and not self.loading_in_process:
                self.load_next(data_to_iterate, last_preloaded_index)
                new_last_preloaded = min(last_preloaded_index + self.preload_size, len(data_to_iterate) - 1)
                last_preloaded_index = new_last_preloaded
                self.need_preload = False
            
            result = self.tokenize_and_prepare_labels_for_audioencoder(example)

            self.mark_file_as_processed(local_path)
            if counter >= self.preload_size - 1 and (counter + 1) % self.preload_size == 0:
                print(f"Worker: {worker_id} Called cleanup at idx:{counter}\n")
                self.cleanup_processed_files(counter, data_to_iterate)
            
            yield result
            