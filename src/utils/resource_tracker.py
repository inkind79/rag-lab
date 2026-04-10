"""
No-op resource tracker to reduce memory overhead.
This replaces the heavy resource tracking implementation.
"""

import time
from contextlib import contextmanager

class NoOpResourceTracker:
    """A no-op implementation of ResourceTracker that does nothing."""
    
    def __init__(self):
        pass
    
    def track_resource(self, resource_type, resource_id, metadata=None):
        """No-op"""
        pass
    
    def untrack_resource(self, resource_type, resource_id):
        """No-op - remove a resource from tracking."""
        pass
    
    def increment_reference(self, resource_id):
        """No-op"""
        pass
    
    def decrement_reference(self, resource_id) -> int:
        """No-op - always return 0"""
        return 0
    
    def get_resource_info(self, resource_id):
        """No-op - return None"""
        return None
    
    def track_connection(self, connection_id):
        """No-op"""
        pass
    
    def untrack_connection(self, connection_id):
        """No-op"""
        pass
    
    def collect_metrics(self):
        """No-op - return empty dict"""
        return {}
    
    def get_resource_report(self):
        """No-op - return empty dict"""
        return {}
    
    def get_all_resources(self):
        """No-op - return empty dict"""
        return {}
    
    def get_metrics(self):
        """No-op - return empty dict"""
        return {}
    
    def cleanup_resource(self, resource_id):
        """No-op"""
        pass
    
    def track_connection(self, connection_id):
        """No-op - track an active connection."""
        pass
    
    def untrack_connection(self, connection_id):
        """No-op - remove a connection from tracking."""
        pass
    
    def release_connection(self, connection_id):
        """No-op"""
        pass
    
    def get_active_connections(self):
        """No-op - return empty dict"""
        return {}
    
    def get_memory_usage(self):
        """No-op - return minimal stats"""
        return {
            "tracked_resources": 0,
            "active_connections": 0,
            "metrics_count": 0
        }
    
    def get_resource_count(self, resource_type):
        """No-op - return 0."""
        return 0
    
    def collect_metrics(self):
        """No-op - return empty metrics."""
        return {
            "memory_rss_mb": 0,
            "memory_vms_mb": 0,
            "gpu_memory_mb": 0,
            "tracked_resources": 0,
            "active_connections": 0,
            "reference_counts": {}
        }
    
    @contextmanager
    def track_operation(self, operation_name):
        """No-op context manager."""
        yield

# Global singleton
_resource_tracker = NoOpResourceTracker()

def get_resource_tracker():
    """Get the global no-op resource tracker."""
    return _resource_tracker

@contextmanager
def track_operation(operation_name):
    """No-op context manager/decorator that works both ways."""
    # When used as context manager
    yield

def collect_metrics():
    """No-op metrics collection."""
    return {
        "tracked_resources": 0,
        "active_connections": 0,
        "metrics_count": 0,
        "operations": {},
        "memory_usage": {
            "tracked_resources": 0,
            "active_connections": 0,
            "metrics_count": 0
        }
    }
