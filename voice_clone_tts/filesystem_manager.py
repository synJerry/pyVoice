import os
import time
import json
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import threading


class FileSystemManager:
    """
    Centralized file system manager with caching to reduce redundant I/O operations.
    """
    
    def __init__(self, cache_ttl: int = 30):
        """
        Initialize the file system manager.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 30)
        """
        self.cache_ttl = cache_ttl
        self._directory_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._file_exists_cache: Dict[str, Tuple[float, bool]] = {}
        self._directory_exists_cache: Dict[str, Tuple[float, bool]] = {}
        self._model_cache: Dict[str, Tuple[float, Dict[str, str]]] = {}
        self._lock = threading.Lock()
        
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.cache_ttl
    
    def _get_directory_mtime(self, path: str) -> float:
        """Get directory modification time, return 0 if not exists."""
        try:
            return os.path.getmtime(path)
        except (OSError, FileNotFoundError):
            return 0
    
    def get_directory_contents(self, path: str, force_refresh: bool = False) -> List[str]:
        """
        Get directory contents with caching.
        
        Args:
            path: Directory path
            force_refresh: Force cache refresh
            
        Returns:
            List of filenames in directory
        """
        with self._lock:
            if not force_refresh and path in self._directory_cache:
                cache_time, contents = self._directory_cache[path]
                if self._is_cache_valid(cache_time):
                    # Also check if directory was modified
                    dir_mtime = self._get_directory_mtime(path)
                    if dir_mtime <= cache_time:
                        return contents.copy()
            
            # Refresh cache
            try:
                if os.path.exists(path):
                    contents = os.listdir(path)
                    self._directory_cache[path] = (time.time(), contents)
                    return contents.copy()
                else:
                    self._directory_cache[path] = (time.time(), [])
                    return []
            except (OSError, PermissionError):
                self._directory_cache[path] = (time.time(), [])
                return []
    
    def file_exists(self, path: str, force_refresh: bool = False) -> bool:
        """
        Check if file exists with caching.
        
        Args:
            path: File path
            force_refresh: Force cache refresh
            
        Returns:
            True if file exists
        """
        with self._lock:
            if not force_refresh and path in self._file_exists_cache:
                cache_time, exists = self._file_exists_cache[path]
                if self._is_cache_valid(cache_time):
                    return exists
            
            # Refresh cache
            exists = os.path.isfile(path)
            self._file_exists_cache[path] = (time.time(), exists)
            return exists
    
    def directory_exists(self, path: str, force_refresh: bool = False) -> bool:
        """
        Check if directory exists with caching.
        
        Args:
            path: Directory path
            force_refresh: Force cache refresh
            
        Returns:
            True if directory exists
        """
        with self._lock:
            if not force_refresh and path in self._directory_exists_cache:
                cache_time, exists = self._directory_exists_cache[path]
                if self._is_cache_valid(cache_time):
                    return exists
            
            # Refresh cache
            exists = os.path.isdir(path)
            self._directory_exists_cache[path] = (time.time(), exists)
            return exists
    
    def batch_file_exists(self, paths: List[str]) -> Dict[str, bool]:
        """
        Check multiple files existence efficiently.
        
        Args:
            paths: List of file paths
            
        Returns:
            Dictionary mapping paths to existence status
        """
        result = {}
        uncached_paths = []
        
        with self._lock:
            # Check cache first
            current_time = time.time()
            for path in paths:
                if path in self._file_exists_cache:
                    cache_time, exists = self._file_exists_cache[path]
                    if self._is_cache_valid(cache_time):
                        result[path] = exists
                        continue
                uncached_paths.append(path)
        
        # Check uncached paths
        for path in uncached_paths:
            exists = os.path.isfile(path)
            with self._lock:
                self._file_exists_cache[path] = (current_time, exists)
            result[path] = exists
        
        return result
    
    def ensure_directory(self, path: str) -> bool:
        """
        Ensure directory exists, create if needed.
        
        Args:
            path: Directory path
            
        Returns:
            True if directory exists or was created successfully
        """
        if self.directory_exists(path):
            return True
        
        try:
            os.makedirs(path, exist_ok=True)
            # Invalidate cache
            with self._lock:
                if path in self._directory_exists_cache:
                    del self._directory_exists_cache[path]
                # Also invalidate parent directory contents cache
                parent = os.path.dirname(path)
                if parent in self._directory_cache:
                    del self._directory_cache[parent]
            return True
        except (OSError, PermissionError):
            return False
    
    def get_model_files(self, model_dir: str, force_refresh: bool = False) -> Dict[str, str]:
        """
        Get voice model files with caching.
        
        Args:
            model_dir: Model directory path
            force_refresh: Force cache refresh
            
        Returns:
            Dictionary mapping speaker IDs to reference audio paths
        """
        with self._lock:
            if not force_refresh and model_dir in self._model_cache:
                cache_time, models = self._model_cache[model_dir]
                if self._is_cache_valid(cache_time):
                    # Check if directory was modified
                    dir_mtime = self._get_directory_mtime(model_dir)
                    if dir_mtime <= cache_time:
                        return models.copy()
        
        # Refresh model cache
        voice_models = {}
        
        if not self.directory_exists(model_dir):
            with self._lock:
                self._model_cache[model_dir] = (time.time(), voice_models)
            return voice_models
        
        contents = self.get_directory_contents(model_dir, force_refresh=True)
        
        for file_name in contents:
            if file_name.endswith("_metadata.json"):
                metadata_path = os.path.join(model_dir, file_name)
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    speaker_id = metadata["speaker_id"]
                    ref_audio_path = os.path.join(model_dir, metadata["reference_audio"])
                    
                    if self.file_exists(ref_audio_path):
                        voice_models[speaker_id] = ref_audio_path
                except (json.JSONDecodeError, KeyError, FileNotFoundError):
                    # Skip invalid metadata files
                    continue
        
        with self._lock:
            self._model_cache[model_dir] = (time.time(), voice_models)
        
        return voice_models.copy()
    
    def has_model_files(self, model_dir: str) -> bool:
        """
        Check if directory contains any valid model files.
        
        Args:
            model_dir: Model directory path
            
        Returns:
            True if model files exist
        """
        if not self.directory_exists(model_dir):
            return False
        
        contents = self.get_directory_contents(model_dir)
        return any(f.endswith("_metadata.json") for f in contents)
    
    def invalidate_cache(self, path: str = None):
        """
        Invalidate cache entries.
        
        Args:
            path: Specific path to invalidate, or None to clear all
        """
        with self._lock:
            if path is None:
                # Clear all caches
                self._directory_cache.clear()
                self._file_exists_cache.clear()
                self._directory_exists_cache.clear()
                self._model_cache.clear()
            else:
                # Clear specific path
                if path in self._directory_cache:
                    del self._directory_cache[path]
                if path in self._file_exists_cache:
                    del self._file_exists_cache[path]
                if path in self._directory_exists_cache:
                    del self._directory_exists_cache[path]
                if path in self._model_cache:
                    del self._model_cache[path]
    
    def cleanup_temp_files(self, paths: List[str]):
        """
        Clean up temporary files safely.
        
        Args:
            paths: List of file paths to remove
        """
        for path in paths:
            try:
                if self.file_exists(path):
                    os.remove(path)
                    # Invalidate cache for removed file
                    with self._lock:
                        if path in self._file_exists_cache:
                            del self._file_exists_cache[path]
            except OSError:
                # Ignore errors during cleanup
                pass
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics for debugging.
        
        Returns:
            Dictionary with cache sizes
        """
        with self._lock:
            return {
                "directory_cache_size": len(self._directory_cache),
                "file_exists_cache_size": len(self._file_exists_cache),
                "directory_exists_cache_size": len(self._directory_exists_cache),
                "model_cache_size": len(self._model_cache)
            }