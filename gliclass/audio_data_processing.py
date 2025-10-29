import json
import random
import queue
import torch
import threading
from torchaudio.transforms import Resample
from torch.utils.data import IterableDataset
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from urllib.parse import urlparse
from pathlib import Path
import time
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, List, Dict, Tuple, Generator, Any, Set
from transformers import PreTrainedTokenizer, FeatureExtractionMixin


class DownloadStatus(Enum):
    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


def parse_s3_path(s3_path: str) -> Tuple[Optional[str], Optional[str]]:
    if not s3_path.startswith('s3://'):
        return None, None

    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def parse_gcs_path(gcs_path: str):
    if not gcs_path.startswith("gs://"):
        return None, None
    path = gcs_path[5:]
    parts = path.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""


def get_local_path(cloud_path: str, local_cache_dir: Union[str, Path]) -> str:
    if cloud_path.startswith('s3://'):
        from urllib.parse import urlparse
        parsed = urlparse(cloud_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
    elif cloud_path.startswith('gs://'):
        path = cloud_path[5:]
        parts = path.split("/", 1)
        bucket = parts[0] if parts else None
        key = parts[1] if len(parts) == 2 else ""
    else:
        return cloud_path
    if bucket is None:
        return cloud_path
    key_parts = key.split("/")
    subset_dir = key_parts[-2]
    filename = key_parts[-1]
    local_path = os.path.join(local_cache_dir, subset_dir, filename)
    return local_path


class JSONLManager:
    def __init__(
        self,
        jsonl_path: str,
        validate_json_file: bool = True,
        buffer_size: int = 8192
    ) -> None:
        self.jsonl_path = jsonl_path
        self.validate_json_file = validate_json_file
        self.buffer_size = buffer_size
        self.invalid_indexes = []

    def get_data_path(self) -> str:
        return self.jsonl_path

    def count_examples(self) -> int:
        count = 0
        invalid_indexes = []

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            remainder = ""
            while True:
                buffer = f.read(self.buffer_size)
                if not buffer:
                    if remainder.strip():
                        if self.validate_json_file:
                            try:
                                json.loads(remainder.strip())
                            except json.JSONDecodeError:
                                invalid_indexes.append(count)
                        count += 1
                    break

                data = remainder + buffer
                lines = data.split('\n')
                remainder = lines[-1]

                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue

                    if self.validate_json_file:
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            invalid_indexes.append(count)
                    count += 1

        if self.validate_json_file and len(invalid_indexes) > 0:
            self.invalid_indexes = invalid_indexes
            print(
                f"{len(invalid_indexes)}/{count} are invalid JSON lines in file: {self.jsonl_path}")
            print(f"Invalid line indexes: {invalid_indexes}")
        else:
            print("All examples are valid!")

        return count

    def get_data_slice_generator(self, start: int = 0, end: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        line_count = 0
        buffer = ""

        with open(self.jsonl_path, 'r', encoding='utf-8', buffering=self.buffer_size) as f:
            while True:
                chunk = f.read(self.buffer_size)
                if not chunk:
                    break

                buffer += chunk
                lines = buffer.split('\n')
                buffer = lines[-1]

                for line in lines[:-1]:
                    if line_count < start:
                        line_count += 1
                        continue

                    if end is not None and line_count >= end:
                        return

                    if line_count in self.invalid_indexes:
                        warnings.warn(
                            f"Skiping line {line_count}; Its invalid json line", UserWarning)
                        line_count += 1
                        continue

                    try:
                        data = json.loads(line.strip())
                        yield data
                    except json.JSONDecodeError:
                        pass
                    line_count += 1

            if buffer and line_count >= start and (end is None or line_count < end):
                try:
                    data = json.loads(buffer.strip())
                    yield data
                except json.JSONDecodeError:
                    pass


class CacheManager:
    GB_KOEF = 1 << 30

    def __init__(self):
        self.local_cache_dir: Optional[Union[str, Path]] = None
        self.preload_size: Optional[int] = None
        self.processed_files: Set[str] = set()
        self.download_lock: Optional[threading.Lock] = None
        self.max_cache_size_mb: Optional[int] = None
        self.s3_manager: Optional['CloudManager'] = None
        self._initialized: bool = False

        self.cleanup_queue = queue.Queue()
        self.cleanup_running = False

    def set_storage_manager(self, s3_manager: 'CloudManager') -> None:
        self.s3_manager = s3_manager

    def init_from_manager(self) -> None:
        if self.s3_manager is None:
            raise ValueError("S3Manager must be set before initialization")
        self.local_cache_dir = self.s3_manager.get_cache_dir()
        self.preload_size = self.s3_manager.get_preload_size()
        self.download_lock = self.s3_manager.get_download_lock()
        self.max_cache_size_mb = self.s3_manager.get_max_cache_size()
        self._initialized = True

    def check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "CacheManager not initialized. Call init_from_s3manager() first")

    def mark_file_as_processed(self, local_path: str | Path) -> None:
        with self.download_lock:
            self.processed_files.add(local_path)

    def _cleanup_batch(self, preloaded_data_start: int, preloaded_data_end: int, data_to_iterate: List[Dict[str, Any]]) -> int:
        files_to_remove = []
        for i in range(preloaded_data_start, min(preloaded_data_end, len(data_to_iterate))):
            s3_path = data_to_iterate[i]['audio_path']
            local_path = get_local_path(s3_path, self.local_cache_dir)
            with self.download_lock:
                if (local_path in self.processed_files and
                    self.s3_manager and
                        local_path not in self.s3_manager.download_futures):
                    files_to_remove.append(local_path)
        removed_count = 0
        for local_path in files_to_remove:
            with self.download_lock:
                is_active_download = local_path in self.s3_manager.download_futures if self.s3_manager else False
            if is_active_download:
                continue
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
                    removed_count += 1
            except Exception as e:
                print(
                    f"failed to remove {Path(local_path).name}: {e}")
            with self.download_lock:
                if self.s3_manager:
                    self.s3_manager.file_status.pop(local_path, None)
                self.processed_files.discard(local_path)
        return removed_count

    def _cleanup_worker(self):
        while True:
            task = self.cleanup_queue.get()
            if task is None:
                break
            try:
                start, end, data = task
                self._cleanup_batch(start, end, data)
            except Exception as e:
                print(f"cleanup error: {e}")

    def _get_dir_size_gb(self, path: Path) -> float:
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total / self.GB_KOEF

    def _should_cleanup_cache(self) -> bool:
        if not Path(self.local_cache_dir).exists():
            return False
        if self.max_cache_size_mb is None:
            return False
        current_size_mb = self._get_dir_size_gb(
            Path(self.local_cache_dir)) * 1024
        exceeds = current_size_mb > self.max_cache_size_mb
        return exceeds

    def cleanup_processed_files(self, counter: int, data_to_iterate: List[Dict[str, Any]]) -> None:

        if not self._should_cleanup_cache():
            return

        if not self.cleanup_running:
            self.cleanup_running = True
            thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            thread.start()

        preloaded_data_start = max(
            ((counter + 1) // self.preload_size - 1) * self.preload_size, 0)
        preloaded_data_end = preloaded_data_start + self.preload_size

        self.cleanup_queue.put(
            (preloaded_data_start, preloaded_data_end, data_to_iterate))

    def get_cache_size_gb(self) -> float:
        self.check_initialized()
        cache = os.listdir(self.local_cache_dir)

        cache_dirs = []
        for object_ in cache:
            path = Path(os.path.join(self.local_cache_dir, object_))
            if path.is_dir():
                cache_dirs.append(path)

        if cache_dirs:
            cache_size_gb = sum(self._get_dir_size_gb(cache_dir)
                                for cache_dir in cache_dirs)
        else:
            cache_size_gb = self._get_dir_size_gb(self.local_cache_dir)
        return cache_size_gb


class CloudManager(ABC):
    def __init__(
        self,
        client: Any,
        local_cache_dir: Union[str, Path],
        preload_size: int = 10,
        max_load_workers: int = 4,
        remaining_preloaded_threshold: int = 3,
        max_cache_size_mb: int = 10240
    ):
        self.client = client
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_mb = max_cache_size_mb
        self.preload_size = preload_size
        self.max_load_workers = max_load_workers
        self.remaining_preloaded_threshold = remaining_preloaded_threshold
        self.download_executor = ThreadPoolExecutor(
            max_workers=max_load_workers,
            thread_name_prefix=self._get_thread_prefix()
        )
        self.download_futures: Dict[str, Any] = {}
        self.file_status: Dict[str, DownloadStatus] = {}
        self.download_lock = threading.Lock()
        self.currently_loading = False
        self.download_stats = {
            'total_requested': 0,
            'completed': 0,
            'failed': 0,
            'cached': 0
        }
        self.cache_manager = CacheManager()
        self.cache_manager.set_storage_manager(self)
        self.cache_manager.init_from_manager()

    @abstractmethod
    def _get_thread_prefix(self) -> str:
        pass

    @abstractmethod
    def _parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        pass

    @abstractmethod
    def _download_file(self, bucket: str, key: str, local_path: str) -> None:
        pass

    def get_cache_dir(self) -> Path:
        return self.local_cache_dir

    def get_preload_size(self) -> int:
        return self.preload_size

    def get_remaining_preloaded_threshold(self) -> int:
        return self.remaining_preloaded_threshold

    def get_download_lock(self) -> threading.Lock:
        return self.download_lock

    def get_load_status(self) -> bool:
        return self.currently_loading

    def get_cache_manager(self):
        return self.cache_manager

    def get_file_status(self, local_path: str) -> DownloadStatus:
        with self.download_lock:
            return self.file_status.get(local_path, DownloadStatus.NOT_STARTED)

    def get_max_cache_size(self) -> int:
        return self.max_cache_size_mb

    def _download_worker(self, cloud_path: str) -> bool:
        bucket, key = self._parse_path(cloud_path)
        if bucket is None:
            return False
        local_path = get_local_path(cloud_path, self.local_cache_dir)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with self.download_lock:
            self.file_status[local_path] = DownloadStatus.DOWNLOADING
        try:
            self._download_file(bucket, key, local_path)
            if os.path.exists(local_path):
                with self.download_lock:
                    self.file_status[local_path] = DownloadStatus.COMPLETED
                    self.download_stats['completed'] += 1
                    self.download_futures.pop(local_path, None)
                return True
        except Exception:
            pass
        with self.download_lock:
            self.file_status[local_path] = DownloadStatus.FAILED
            self.download_stats['failed'] += 1
        return False

    def _start_download(self, cloud_path: str) -> None:
        local_path = get_local_path(cloud_path, self.local_cache_dir)
        with self.download_lock:
            self.file_status[local_path] = DownloadStatus.NOT_STARTED
            self.download_stats['total_requested'] += 1
        future = self.download_executor.submit(
            self._download_worker, cloud_path)
        with self.download_lock:
            self.download_futures[local_path] = future

    def load_next(self, data_to_iterate: List[Dict], last_preloaded_index: int, dynamic_size: Optional[int] = None) -> None:
        size = dynamic_size if dynamic_size is not None else self.preload_size
        start = last_preloaded_index + 1
        end = min(start + size, len(data_to_iterate))
        if start >= end:
            return
        with self.download_lock:
            self.currently_loading = True
        for i in range(start, end):
            cloud_path = data_to_iterate[i]['audio_path']
            local_path = get_local_path(cloud_path, self.local_cache_dir)
            with self.download_lock:
                status = self.file_status.get(
                    local_path, DownloadStatus.NOT_STARTED)
            if status in [DownloadStatus.COMPLETED, DownloadStatus.DOWNLOADING]:
                if status == DownloadStatus.COMPLETED:
                    with self.download_lock:
                        self.download_stats['cached'] += 1
                continue
            self._start_download(cloud_path)
        with self.download_lock:
            self.currently_loading = False

    def ensure_loaded(self, cloud_path: str, max_wait_time: int = 30, max_retries: int = 3, retry_delay: float = 1.0) -> Union[str, bool]:
        bucket, key = self._parse_path(cloud_path)
        if bucket is None:
            return cloud_path
        local_path = get_local_path(cloud_path, self.local_cache_dir)
        for attempt in range(1, max_retries + 1):
            with self.download_lock:
                status = self.file_status.get(
                    local_path, DownloadStatus.NOT_STARTED)
            if os.path.exists(local_path) and status == DownloadStatus.COMPLETED:
                return local_path
            if status in [DownloadStatus.NOT_STARTED, DownloadStatus.FAILED]:
                self._start_download(cloud_path)
                with self.download_lock:
                    status = self.file_status.get(
                        local_path, DownloadStatus.DOWNLOADING)
            if status == DownloadStatus.DOWNLOADING:
                waited = 0.0
                while waited < max_wait_time:
                    with self.download_lock:
                        status = self.file_status.get(
                            local_path, DownloadStatus.DOWNLOADING)
                    if status == DownloadStatus.COMPLETED:
                        break
                    if status == DownloadStatus.FAILED:
                        break
                    time.sleep(0.1)
                    waited += 0.1
            if status == DownloadStatus.COMPLETED and os.path.exists(local_path):
                with self.download_lock:
                    self.file_status[local_path] = DownloadStatus.COMPLETED
                return local_path
            time.sleep(retry_delay * attempt)
        with self.download_lock:
            self.file_status[local_path] = DownloadStatus.FAILED
        return False


class GCSManager(CloudManager):
    def __init__(
        self,
        gcs_client,
        local_cache_dir: Union[str, Path],
        preload_size: int = 10,
        max_load_workers: int = 4,
        remaining_preloaded_threshold: int = 3,
        max_cache_size_mb: int = 10240
    ):
        super().__init__(
            client=gcs_client,
            local_cache_dir=local_cache_dir,
            preload_size=preload_size,
            max_load_workers=max_load_workers,
            remaining_preloaded_threshold=remaining_preloaded_threshold,
            max_cache_size_mb=max_cache_size_mb,
        )

    def _get_thread_prefix(self) -> str:
        return "gcs_downloader"

    def _parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        return parse_gcs_path(path)

    def _download_file(self, bucket_name: str, blob_name: str, local_path: str) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)


class S3Manager(CloudManager):
    def __init__(
        self,
        s3_client,
        local_cache_dir: Union[str, Path],
        preload_size: int = 10,
        max_load_workers: int = 4,
        remaining_preloaded_threshold: int = 3,
        max_cache_size_mb: int = 10240,
    ):
        super().__init__(
            client=s3_client,
            local_cache_dir=local_cache_dir,
            preload_size=preload_size,
            max_load_workers=max_load_workers,
            remaining_preloaded_threshold=remaining_preloaded_threshold,
            max_cache_size_mb=max_cache_size_mb,
        )

    def _get_thread_prefix(self) -> str:
        return "s3_downloader"

    def _parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        return parse_s3_path(path)

    def _download_file(self, bucket: str, key: str, local_path: str) -> None:
        self.client.download_file(bucket, key, local_path)


class GLiClassAudioDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: str | Path,
        cloud_manager: CloudManager,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        problem_type: str = 'multi_label_classification',
        architecture_type: str = 'audio-encoder',
        audio_features_extractor: FeatureExtractionMixin = None,
        sampling_rate: int = 16000,
        max_duration_s: int = 15,
        shuffle_labels: bool = True,
        **json_manger_kwargs
    ) -> None:
        if architecture_type != 'audio-encoder':
            raise ValueError(
                "This class was specifecly created for 'audio-encoder' arch")
        if audio_features_extractor is None:
            raise ValueError("audio_features_extractor was not provided")
        if sampling_rate is None or max_duration_s is None:
            raise ValueError(
                "When using audio_features_extractor you must specify sampling_rate и max_duration_s")

        self.cloud_manager = cloud_manager
        self.preload_size = self.cloud_manager.get_preload_size()
        self.remaining_preloaded_threshold = self.cloud_manager.get_remaining_preloaded_threshold()
        self.local_cache_dir = self.cloud_manager.get_cache_dir()
        self.need_preload = False

        self.cache_manager = cloud_manager.get_cache_manager()
        self.tokenizer = tokenizer
        self.audio_features_extractor = audio_features_extractor
        self.max_length = max_length
        self.dataset_path = dataset_path
        self.problem_type = problem_type
        self.shuffle_labels = shuffle_labels

        self.jsonl_manager = JSONLManager(
            jsonl_path=dataset_path, **json_manger_kwargs)
        self.num_examples = 5000000

        self.sampling_rate = sampling_rate
        self.max_duration_s = max_duration_s
        self.max_duration_samples = self.sampling_rate * self.max_duration_s
        print(
            f"Audio parameters: sampling_rate={self.sampling_rate}, max_duration={self.max_duration_s}s, max_samples={self.max_duration_samples}")

    def get_num_examples(self) -> int:
        return self.num_examples

    def prepare_labels(self, example: Dict[str, Any], label2idx: Dict[str, int], problem_type: str, worker_id) -> torch.Tensor:
        if problem_type == 'single_label_classification':
            labels = label2idx[example['true_labels'][0]]
        elif problem_type == 'multi_label_classification':
            if isinstance(example['true_labels'], dict):
                labels = [example['true_labels'].get(
                    label, 0.) for label in example['all_labels']]
            else:
                labels = [1. if label in example['true_labels']
                          else 0. for label in example['all_labels']]
        else:
            raise NotImplementedError(f"{problem_type} is not implemented.")
        return torch.tensor(labels)

    def prepare_prompt(self, example: Dict[str, Any]) -> List[str]:
        prompt_texts = [
            f"<<LABEL>>{str(label)}" for label in example['all_labels']]
        prompt_texts.append('<<SEP>>')
        return prompt_texts

    def prepare_audio(self, audio_array: Union[np.ndarray, torch.Tensor, List], audio_sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(audio_array, np.ndarray):
            audio_array = torch.from_numpy(audio_array).float()
        elif not isinstance(audio_array, torch.Tensor):
            audio_array = torch.tensor(audio_array, dtype=torch.float32)
        else:
            audio_array = audio_array.float()

        if audio_sr != self.sampling_rate:
            audio_array = Resample(
                audio_sr, new_freq=self.sampling_rate)(audio_array)

        audio_inputs = self.audio_features_extractor(
            audio_array,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_duration_samples
        )
        return audio_inputs["input_values"], audio_inputs["attention_mask"]

    def tokenize(self, texts) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )

    def tokenize_and_prepare_labels_for_audioencoder(self, example: Dict[str, Any], worker_id) -> Dict[str, torch.Tensor]:
        if self.shuffle_labels:
            random.shuffle(example['all_labels'])
        input_text = ''.join(self.prepare_prompt(example) + ['<<AUDIO>>'])
        label2idx = {label: idx for idx,
                     label in enumerate(example['all_labels'])}

        tokenized_inputs = self.tokenize(input_text)
        tokenized_inputs['labels'] = self.prepare_labels(
            example, label2idx, self.problem_type, worker_id)

        audio_data = torch.load(example['audio_path'], weights_only=False)
        audio_sr = example["sample_rate"]
        tokenized_inputs["input_audio_features"], tokenized_inputs["audio_attention_mask"] = self.prepare_audio(
            audio_data, audio_sr)
        return tokenized_inputs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            per_process = int(np.ceil(self.num_examples / float(world_size)))
            process_start = rank * per_process
            process_end = min(process_start + per_process, self.num_examples)
        else:
            process_start, process_end, rank = 0, self.num_examples, 0
        process_data_size = process_end - process_start
        if worker_info is None:
            worker_id = "main"
            worker_start, worker_end = process_start, process_end
        else:
            per_worker = int(np.ceil(process_data_size /
                             float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_start = process_start + (worker_id * per_worker)
            worker_end = min(worker_start + per_worker, process_end)
        data_to_iterate = list(
            self.jsonl_manager.get_data_slice_generator(worker_start, worker_end))
        self.cloud_manager.load_next(data_to_iterate, -1)
        last_preloaded_index = min(
            self.preload_size - 1, len(data_to_iterate) - 1)
        BUFFER_SIZE = 2
        for counter, example in enumerate(data_to_iterate):
            cloud_path = example['audio_path']
            if not self.cloud_manager.ensure_loaded(cloud_path):
                warnings.warn(
                    f"Skipping example {counter} as failed to load its audio file", UserWarning)
                continue
            local_path = get_local_path(cloud_path, self.local_cache_dir)
            example['audio_path'] = local_path
            remaining_preloaded = last_preloaded_index - counter
            dynamic_preload_size = self.preload_size
            if remaining_preloaded <= self.remaining_preloaded_threshold and not self.cloud_manager.get_load_status():
                self.need_preload = True
            if self.need_preload and not self.cloud_manager.get_load_status():
                cleanup_end_index = counter
                files_freed = 0
                if cleanup_end_index > 0:
                    files_freed = self.cache_manager._cleanup_batch(0, cleanup_end_index, data_to_iterate)
                    if files_freed > 0:
                        dynamic_preload_size = max(1, files_freed - BUFFER_SIZE)
                    elif self.cache_manager._should_cleanup_cache():
                        self.need_preload = False
                        continue
                self.cloud_manager.load_next(
                    data_to_iterate,
                    last_preloaded_index,
                    dynamic_size=dynamic_preload_size
                )
                last_preloaded_index = min(
                    last_preloaded_index + dynamic_preload_size, len(data_to_iterate) - 1)
                self.need_preload = False
            result = self.tokenize_and_prepare_labels_for_audioencoder(
                example, worker_id)
            self.cache_manager.mark_file_as_processed(local_path)
            yield result
