"""
Simplified resource lifecycle manager to reduce memory overhead.
"""

class NoOpResourceHandle:
    """A no-op resource handle."""
    
    def __init__(self, resource_id, resource, cleanup_fn=None):
        self.resource_id = resource_id
        self.resource = resource
        self._cleanup_fn = cleanup_fn
    
    def get(self):
        """Get the resource."""
        return self.resource
    
    def release(self):
        """Release the resource by calling cleanup if provided."""
        if self._cleanup_fn and self.resource:
            try:
                self._cleanup_fn(self.resource)
            except:
                pass
        self.resource = None

class SimplifiedLifecycleManager:
    """A simplified lifecycle manager without the overhead."""
    
    def __init__(self):
        self._resources = {}
    
    def register_resource(self, resource_id, resource, cleanup_fn=None):
        """Register a resource - simplified version."""
        handle = NoOpResourceHandle(resource_id, resource, cleanup_fn)
        self._resources[resource_id] = handle
        return handle
    
    def release_resource(self, resource_id):
        """Release a resource."""
        if resource_id in self._resources:
            handle = self._resources.pop(resource_id)
            handle.release()
    
    def acquire_resource(self, resource_id, resource, cleanup_fn=None):
        """Acquire a resource - alias for register_resource."""
        return self.register_resource(resource_id, resource, cleanup_fn)
    
    def get_resource(self, resource_id):
        """Get a resource if it exists."""
        handle = self._resources.get(resource_id)
        return handle.get() if handle else None
    
    def cleanup_all(self):
        """Release all resources."""
        for resource_id in list(self._resources.keys()):
            self.release_resource(resource_id)

# Global singleton
_lifecycle_manager = SimplifiedLifecycleManager()

def get_lifecycle_manager():
    """Get the global lifecycle manager."""
    return _lifecycle_manager

# No-op decorator
def managed_resource(resource_id, cleanup_fn=None):
    """No-op context manager."""
    class NoOpContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return NoOpContext()

