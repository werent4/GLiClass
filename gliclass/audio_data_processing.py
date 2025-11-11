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
from tqdm import tqdm


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
    def __init__(self, jsonl_path: str, validate_json_file: bool = True, buffer_size: int = 8192) -> None:
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
        self.cloud_manager: Optional['CloudManager'] = None
        self._initialized: bool = False
        self.local_file_sizes: Dict[str, int] = {}
        self.cleanup_executor: Optional[ThreadPoolExecutor] = None

    def set_storage_manager(self, cloud_manager: 'CloudManager') -> None:
        self.cloud_manager = cloud_manager

    def init_from_cloud_manager(self) -> None:
        if self.cloud_manager is None:
            raise ValueError("CloudManager must be set before initialization")
        
        self.local_cache_dir = self.cloud_manager.get_cache_dir()
        self.preload_size = self.cloud_manager.get_preload_size()
        self.download_lock = self.cloud_manager.get_download_lock()
        self.max_cache_size_mb = self.cloud_manager.get_max_cache_size()
        self.cleanup_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cache_cleanup"
        )
        self._initialized = True

    def check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("CacheManager not initialized. Call init_from_cloud_manager() first")

    def mark_file_as_processed(self, local_path: str | Path) -> None:
        with self.download_lock:
            self.processed_files.add(str(local_path))

    def register_file_size(self, local_path: str | Path, size_bytes: int) -> None:
        with self.download_lock:
            self.local_file_sizes[str(local_path)] = int(size_bytes)

    def _physical_cleanup_files(self, files_to_remove: List[str]) -> int:
        total_freed = 0
        
        for local_path in files_to_remove:
            with self.download_lock:
                file_size = self.local_file_sizes.pop(str(local_path), 0)
                self.processed_files.discard(str(local_path))
                if self.cloud_manager:
                    self.cloud_manager.file_status.pop(str(local_path), None)
            
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
                    total_freed += file_size
            except Exception:
                pass
        
        return total_freed

    def cleanup_processed_range_bytes(self, preloaded_data_start: int, preloaded_data_end: int, 
                                     data_to_iterate: List[Dict[str, Any]]) -> int:
        files_to_remove = []
        
        for i in range(preloaded_data_start, min(preloaded_data_end, len(data_to_iterate))):
            cloud_path = data_to_iterate[i]['audio_path']
            local_path = get_local_path(cloud_path, self.local_cache_dir)

            with self.download_lock:
                is_processed = str(local_path) in self.processed_files
                is_downloading = self.cloud_manager and str(local_path) in self.cloud_manager.download_futures
                file_size = self.local_file_sizes.get(str(local_path), 0)

            if is_processed and not is_downloading and file_size > 0:
                files_to_remove.append(str(local_path))
        
        if len(files_to_remove) > 0:
            future = self.cleanup_executor.submit(self._physical_cleanup_files, files_to_remove)
            logical_freed = future.result()
            return logical_freed
        
        return 0

    def get_used_bytes(self) -> int:
        self.check_initialized()
        with self.download_lock:
            total = sum(self.local_file_sizes.values())
        return int(total)

    def get_max_cache_bytes(self) -> int:
        if self.max_cache_size_mb is None:
            return int(1e18)
        return int(self.max_cache_size_mb * 1024 * 1024)

    def get_free_space_bytes(self) -> int:
        return max(0, self.get_max_cache_bytes() - self.get_used_bytes())

    def unregister_file_size(self, local_path: str | Path) -> None:
        key = str(local_path)
        
        file_size = 0
        with self.download_lock:
            file_size = self.local_file_sizes.pop(key, 0)
        
        try:
            if os.path.exists(key):
                os.remove(key)
        except Exception:
            with self.download_lock:
                if os.path.exists(key):
                    self.local_file_sizes[key] = file_size


class CloudManager(ABC):
    def __init__(
        self,
        client: Any,
        local_cache_dir: Union[str, Path],
        preload_size: int = 10,
        max_load_workers: int = 4,
        remaining_preloaded_threshold: int = 3,
        max_cache_size_mb: int = 10240,
        safety_margin_percent: float = 0.15
    ):
        self.client = client
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_cache_size_mb = max_cache_size_mb
        self.preload_size = preload_size
        self.max_load_workers = max_load_workers
        self.remaining_preloaded_threshold = remaining_preloaded_threshold
        self.safety_margin_percent = safety_margin_percent
        self.remote_file_sizes: dict[str, int] = {}
        
        self.download_executor = ThreadPoolExecutor(
            max_workers=max_load_workers,
            thread_name_prefix=self._get_thread_prefix()
        )
        self.pipeline_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{self._get_thread_prefix()}_pipeline"
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
        self.cache_manager.init_from_cloud_manager()

    @abstractmethod
    def _get_thread_prefix(self) -> str:
        pass

    @abstractmethod
    def _parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        pass

    @abstractmethod
    def _download_file(self, bucket: str, key: str, local_path: str) -> None:
        pass

    @abstractmethod
    def _get_remote_file_size(self, cloud_path: str) -> int:
        pass

    def estimate_next_batch_size_bytes(self, data_to_iterate: List[Dict], last_index: int,
                                      dynamic_size: Optional[int] = None) -> int:
        size = dynamic_size if dynamic_size is not None else self.preload_size
        start = last_index + 1
        end = min(start + size, len(data_to_iterate))
        
        if start >= end:
            return 0
        
        total_bytes = 0
        
        for i in range(start, end):
            cloud_path = data_to_iterate[i]['audio_path']
            
            if cloud_path in self.remote_file_sizes:
                total_bytes += self.remote_file_sizes[cloud_path]
                continue
            
            file_size = self._get_remote_file_size(cloud_path)
            
            if file_size > 0:
                self.remote_file_sizes[cloud_path] = file_size
            
            total_bytes += file_size
        
        return total_bytes

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
            self.file_status[str(local_path)] = DownloadStatus.DOWNLOADING
        
        try:
            estimated_size = self.remote_file_sizes.get(cloud_path, 0)
            free_space = self.cache_manager.get_free_space_bytes()
            
            if estimated_size > free_space:
                with self.download_lock:
                    self.file_status[str(local_path)] = DownloadStatus.FAILED
                    self.download_stats['failed'] += 1
                return False
            
            self._download_file(bucket, key, local_path)
            
            if os.path.exists(local_path):
                with self.download_lock:
                    self.file_status[str(local_path)] = DownloadStatus.COMPLETED
                    self.download_stats['completed'] += 1
                    self.download_futures.pop(str(local_path), None)
                return True
                
        except Exception:
            pass
        
        with self.download_lock:
            self.file_status[str(local_path)] = DownloadStatus.FAILED
            self.download_stats['failed'] += 1
        return False

    def _start_download(self, cloud_path: str) -> None:
        local_path = get_local_path(cloud_path, self.local_cache_dir)
        with self.download_lock:
            self.file_status[str(local_path)] = DownloadStatus.DOWNLOADING
            self.download_stats['total_requested'] += 1
        future = self.download_executor.submit(self._download_worker, cloud_path)
        with self.download_lock:
            self.download_futures[str(local_path)] = future

    def _prepare_batch_downloads(self, data_to_iterate: List[Dict], start: int, 
                                batch_size: int) -> int:
        self.estimate_next_batch_size_bytes(data_to_iterate, start - 1, dynamic_size=batch_size)
        
        free_bytes = self.cache_manager.get_free_space_bytes()
        max_cache_bytes = self.cache_manager.get_max_cache_bytes()
        safety_margin = int(self.safety_margin_percent * max_cache_bytes)
        available_bytes = max(0, free_bytes - safety_margin)
        
        to_download = []
        total_bytes = 0
        for i in range(start, min(start + batch_size, len(data_to_iterate))):
            cloud_path = data_to_iterate[i]['audio_path']
            file_size = self.remote_file_sizes.get(cloud_path, 0)
            
            if total_bytes + file_size <= available_bytes:
                to_download.append((i, cloud_path, file_size))
                total_bytes += file_size
            else:
                break
        
        for i, cloud_path, file_size in to_download:
            local_path = get_local_path(cloud_path, self.local_cache_dir)
            with self.download_lock:
                status = self.file_status.get(str(local_path), DownloadStatus.NOT_STARTED)
            
            if status not in [DownloadStatus.COMPLETED, DownloadStatus.DOWNLOADING]:
                self._start_download(cloud_path)
        
        return len(to_download)

    def _async_pipeline_load(self, data_to_iterate: List[Dict], start: int, 
                            prefetch_size: int) -> None:
        with self.download_lock:
            if self.currently_loading:
                return
            self.currently_loading = True
        
        try:
            self._prepare_batch_downloads(data_to_iterate, start, prefetch_size)
        finally:
            with self.download_lock:
                self.currently_loading = False

    def load_initial_batch(self, data_to_iterate: List[Dict], batch_size: Optional[int] = None) -> int:
        initial_check_size = min(batch_size or self.preload_size, len(data_to_iterate))
        if initial_check_size == 0:
            return -1
        
        with self.download_lock:
            self.currently_loading = True
        
        try:
            loaded_count = self._prepare_batch_downloads(data_to_iterate, 0, initial_check_size)
            return loaded_count - 1 if loaded_count > 0 else -1
        finally:
            with self.download_lock:
                self.currently_loading = False

    def load_next(self, data_to_iterate: List[Dict], last_preloaded_index: int, 
                 dynamic_size: Optional[int] = None) -> None:
        start = last_preloaded_index + 1
        if start >= len(data_to_iterate):
            return
        
        prefetch_size = dynamic_size or self.preload_size
        prefetch_size = min(prefetch_size, len(data_to_iterate) - start)
        
        self.pipeline_executor.submit(
            self._async_pipeline_load,
            data_to_iterate, start, prefetch_size
        )

    def _check_and_start_download(self, cloud_path: str, local_path: str) -> DownloadStatus:
        with self.download_lock:
            status = self.file_status.get(str(local_path), DownloadStatus.NOT_STARTED)
        
        if os.path.exists(local_path) and status == DownloadStatus.COMPLETED:
            return DownloadStatus.COMPLETED
        
        if status in [DownloadStatus.NOT_STARTED, DownloadStatus.FAILED]:
            self._start_download(cloud_path)
            return DownloadStatus.DOWNLOADING
        
        return status

    def _wait_for_download(self, local_path: str, max_wait_time: int) -> DownloadStatus:
        waited = 0.0
        while waited < max_wait_time:
            with self.download_lock:
                status = self.file_status.get(str(local_path), DownloadStatus.DOWNLOADING)
            
            if status in [DownloadStatus.COMPLETED, DownloadStatus.FAILED]:
                return status
            
            time.sleep(0.1)
            waited += 0.1
        
        return DownloadStatus.DOWNLOADING

    def ensure_loaded(self, cloud_path: str, max_wait_time: int = 30, 
                     max_retries: int = 3, retry_delay: float = 1.0) -> Union[str, bool]:
        bucket, key = self._parse_path(cloud_path)
        if bucket is None:
            return cloud_path
        
        local_path = get_local_path(cloud_path, self.local_cache_dir)
        
        for attempt in range(1, max_retries + 1):
            status = self._check_and_start_download(cloud_path, local_path)
            
            if status == DownloadStatus.COMPLETED:
                return local_path
            
            if status == DownloadStatus.DOWNLOADING:
                status = self._wait_for_download(local_path, max_wait_time)
                if status == DownloadStatus.COMPLETED:
                    return local_path
            
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
        
        with self.download_lock:
            self.file_status[str(local_path)] = DownloadStatus.FAILED
        return False


class GCSManager(CloudManager):
    def __init__(
        self,
        gcs_client,
        local_cache_dir: Union[str, Path],
        preload_size: int = 10,
        max_load_workers: int = 4,
        remaining_preloaded_threshold: int = 3,
        max_cache_size_mb: int = 10240,
        safety_margin_percent: float = 0.15
    ):
        super().__init__(
            client=gcs_client,
            local_cache_dir=local_cache_dir,
            preload_size=preload_size,
            max_load_workers=max_load_workers,
            remaining_preloaded_threshold=remaining_preloaded_threshold,
            max_cache_size_mb=max_cache_size_mb,
            safety_margin_percent=safety_margin_percent,
        )

    def _get_thread_prefix(self) -> str:
        return "gcs_downloader"

    def _parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        return parse_gcs_path(path)

    def _get_remote_file_size(self, cloud_path: str) -> int:
        bucket_name, blob_name = self._parse_path(cloud_path)
        if bucket_name is None:
            return 0
        
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.reload()
            return blob.size or 0
        except Exception:
            return 0

    def _download_file(self, bucket_name: str, blob_name: str, local_path: str) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        self.cache_manager.register_file_size(local_path, blob.size or os.path.getsize(local_path))

    def _get_file_size_with_index(self, idx_path_tuple: Tuple[int, str]) -> Tuple[str, int]:
        idx, cloud_path = idx_path_tuple
        file_size = self._get_remote_file_size(cloud_path)
        return (cloud_path, file_size)

    def estimate_next_batch_size_bytes(self, data_to_iterate: List[Dict], last_index: int,
                                      dynamic_size: Optional[int] = None) -> int:
        size = dynamic_size if dynamic_size is not None else self.preload_size
        start = last_index + 1
        end = min(start + size, len(data_to_iterate))
        
        if start >= end:
            return 0
        
        files_to_check = []
        cached_bytes = 0
        
        for i in range(start, end):
            cloud_path = data_to_iterate[i]['audio_path']
            if cloud_path in self.remote_file_sizes:
                cached_bytes += self.remote_file_sizes[cloud_path]
            else:
                files_to_check.append((i, cloud_path))
        
        if files_to_check:
            max_workers = min(8, len(files_to_check))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gcs_size_checker") as executor:
                futures = [executor.submit(self._get_file_size_with_index, item) for item in files_to_check]
                results = [f.result() for f in futures]
            
            fetched_bytes = 0
            for cloud_path, file_size in results:
                if file_size > 0:
                    self.remote_file_sizes[cloud_path] = file_size
                    fetched_bytes += file_size
        else:
            fetched_bytes = 0
        
        total_bytes = cached_bytes + fetched_bytes
        return total_bytes


class S3Manager(CloudManager):
    def __init__(
        self,
        s3_client,
        local_cache_dir: Union[str, Path],
        preload_size: int = 10,
        max_load_workers: int = 4,
        remaining_preloaded_threshold: int = 3,
        max_cache_size_mb: int = 10240,
        safety_margin_percent: float = 0.15,
    ):
        super().__init__(
            client=s3_client,
            local_cache_dir=local_cache_dir,
            preload_size=preload_size,
            max_load_workers=max_load_workers,
            remaining_preloaded_threshold=remaining_preloaded_threshold,
            max_cache_size_mb=max_cache_size_mb,
            safety_margin_percent=safety_margin_percent,
        )

    def _get_thread_prefix(self) -> str:
        return "s3_downloader"

    def _parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        return parse_s3_path(path)

    def _download_file(self, bucket: str, key: str, local_path: str) -> None:
        self.client.download_file(bucket, key, local_path)

    def _get_remote_file_size(self, cloud_path: str) -> int:
        bucket, key = self._parse_path(cloud_path)
        if bucket is None:
            return 0
        
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
            return response.get('ContentLength', 0)
        except Exception:
            return 0

    def _get_file_size_by_index(self, data_to_iterate: list[dict], i: int) -> int:
        cloud_path = data_to_iterate[i]['audio_path']
        bucket, key = self._parse_path(cloud_path)
        if bucket is None:
            return 0
        
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
            return response.get('ContentLength', 0)
        except Exception:
            return 0

    def estimate_next_batch_size_bytes(self, data_to_iterate: list[dict], last_index: int,
                                      dynamic_size: Optional[int] = None) -> int:
        size = dynamic_size if dynamic_size is not None else self.preload_size
        start = last_index + 1
        end = min(start + size, len(data_to_iterate))
        
        if start >= end:
            return 0
        
        futures = [
            self.download_executor.submit(self._get_file_size_by_index, data_to_iterate, i) 
            for i in range(start, end)
        ]
        sizes = [f.result() for f in futures]
        
        return sum(sizes)


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
        rank = None,
        world_size = -1,
        **json_manger_kwargs
    ) -> None:
        if architecture_type != 'audio-encoder':
            raise ValueError("This class was specifically created for 'audio-encoder' arch")
        if audio_features_extractor is None:
            raise ValueError("audio_features_extractor was not provided")
        if sampling_rate is None or max_duration_s is None:
            raise ValueError("When using audio_features_extractor you must specify sampling_rate and max_duration_s")

        self.rank = rank
        self.world_size = world_size
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

        self.jsonl_manager = JSONLManager(jsonl_path=dataset_path, **json_manger_kwargs)
        self.num_examples = self.jsonl_manager.count_examples()

        self.sampling_rate = sampling_rate
        self.max_duration_s = max_duration_s
        self.max_duration_samples = self.sampling_rate * self.max_duration_s

    def get_num_examples(self) -> int:
        return self.num_examples

    def prepare_labels(self, example: Dict[str, Any], label2idx: Dict[str, int], 
                      problem_type: str, worker_id) -> torch.Tensor:
        if problem_type == 'single_label_classification':
            labels = label2idx[example['true_labels'][0]]
        elif problem_type == 'multi_label_classification':
            if isinstance(example['true_labels'], dict):
                labels = [example['true_labels'].get(label, 0.) for label in example['all_labels']]
            else:
                labels = [1. if label in example['true_labels'] else 0. for label in example['all_labels']]
        else:
            raise NotImplementedError(f"{problem_type} is not implemented.")
        
        return torch.tensor(labels)

    def prepare_prompt(self, example: Dict[str, Any]) -> List[str]:
        prompt_texts = [f"<<LABEL>>{str(label)}" for label in example['all_labels']]
        prompt_texts.append('<<SEP>>')
        return prompt_texts

    def prepare_audio(self, audio_array: Union[np.ndarray, torch.Tensor, List], 
                     audio_sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(audio_array, np.ndarray):
            audio_array = torch.from_numpy(audio_array).float()
        elif not isinstance(audio_array, torch.Tensor):
            audio_array = torch.tensor(audio_array, dtype=torch.float32)
        else:
            audio_array = audio_array.float()

        if audio_sr != self.sampling_rate:
            audio_array = Resample(audio_sr, new_freq=self.sampling_rate)(audio_array)

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
            texts, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )

    def tokenize_and_prepare_labels_for_audioencoder(self, example: Dict[str, Any], 
                                                worker_id) -> Dict[str, torch.Tensor]:
        if self.shuffle_labels:
            random.shuffle(example['all_labels'])
        
        input_text = ''.join(self.prepare_prompt(example) + ['<<AUDIO>>'])
        label2idx = {label: idx for idx, label in enumerate(example['all_labels'])}

        original_len = len(self.tokenizer.encode(input_text, add_special_tokens=True))
        
        tokenized_inputs = self.tokenize(input_text)
        tokenized_inputs['original_seq_len'] = original_len
        tokenized_inputs['labels'] = self.prepare_labels(example, label2idx, self.problem_type, worker_id)

        audio_data = torch.load(example['audio_path'], weights_only=False)
        audio_sr = example["sample_rate"]
        tokenized_inputs["input_audio_features"], tokenized_inputs["audio_attention_mask"] = self.prepare_audio(
            audio_data, audio_sr)
        
        return tokenized_inputs

    def __len__(self):
        if self.world_size > 1:
            return np.ceil(self.num_examples / self.world_size)
        return self.num_examples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if self.rank is not None and self.world_size > 1:
            per_process = int(np.ceil(self.num_examples / float(self.world_size)))
            process_start = self.rank * per_process
            process_end = min(process_start + per_process, self.num_examples)
        else:
            process_start = 0
            process_end = self.num_examples
        
        process_data_size = process_end - process_start
        
        if worker_info is None:
            worker_id = "main"
            worker_start = process_start
            worker_end = process_end
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(np.ceil(process_data_size / float(num_workers)))
            worker_start = process_start + (worker_id * per_worker) 
            worker_end = min(worker_start + per_worker, process_end)

        data_to_iterate = list(self.jsonl_manager.get_data_slice_generator(worker_start, worker_end))
        
        max_cache_bytes = self.cache_manager.get_max_cache_bytes()
        safety_margin = int(self.cloud_manager.safety_margin_percent * max_cache_bytes)
        
        last_preloaded_index = self.cloud_manager.load_initial_batch(data_to_iterate)
        
        for counter, example in enumerate(data_to_iterate):
            cloud_path = example['audio_path']
            local_path = get_local_path(cloud_path, self.local_cache_dir)
            
            ensure_result = self.cloud_manager.ensure_loaded(cloud_path)
            if not ensure_result or not os.path.exists(local_path):
                warnings.warn(f"Skipping {counter}: {Path(local_path).name}", UserWarning)
                continue
            example['audio_path'] = local_path
            result = self.tokenize_and_prepare_labels_for_audioencoder(example, self.rank)
            self.cache_manager.mark_file_as_processed(local_path)
            
            remaining_preloaded = last_preloaded_index - counter
            if remaining_preloaded <= self.remaining_preloaded_threshold:
                if not self.cloud_manager.get_load_status():
                    self.need_preload = True
            
            if self.need_preload and not self.cloud_manager.get_load_status():
                cleanup_start = max(last_preloaded_index + 1 - self.preload_size, 0)
                cleanup_end = last_preloaded_index + 1
                
                freed = self.cache_manager.cleanup_processed_range_bytes(
                    cleanup_start, cleanup_end, data_to_iterate
                )
                
                start_idx = last_preloaded_index + 1
                end_idx = min(start_idx + self.preload_size, len(data_to_iterate))
                
                self.cloud_manager.estimate_next_batch_size_bytes(
                    data_to_iterate, last_preloaded_index, dynamic_size=self.preload_size
                )
                
                free_bytes = self.cache_manager.get_free_space_bytes()
                available_bytes_for_new = max(0, free_bytes - safety_margin)
                
                actual_loaded = 0
                total_bytes_to_load = 0
                
                for i in range(start_idx, end_idx):
                    cloud_path_check = data_to_iterate[i]['audio_path']
                    file_size = self.cloud_manager.remote_file_sizes.get(cloud_path_check, 0)
                    
                    if total_bytes_to_load + file_size <= available_bytes_for_new:
                        total_bytes_to_load += file_size
                        actual_loaded += 1
                    else:
                        break
                
                if actual_loaded == 0 and end_idx > start_idx:
                    actual_loaded = 1
                    total_bytes_to_load = self.cloud_manager.remote_file_sizes.get(
                        data_to_iterate[start_idx]['audio_path'], 0)
                
                old_index = last_preloaded_index
                last_preloaded_index = min(
                    last_preloaded_index + actual_loaded,
                    len(data_to_iterate) - 1
                )
                
                self.cloud_manager.load_next(data_to_iterate, old_index, dynamic_size=actual_loaded)
                self.need_preload = False
            yield result