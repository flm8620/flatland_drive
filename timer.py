"""
Performance timing utilities for IsaacLab environments with hierarchical timing support.
"""

import time
import torch
from typing import Dict, Optional, List, Set
import threading


class NoOpTimerNode:
    """A no-operation timer node that does nothing when timing is disabled."""
    
    def __init__(self, name: str, manager: 'HierarchicalTimerManager'):
        self.name = name
        self.manager = manager
        self.enabled = False
        
    def get_or_create_child(self, name: str) -> 'NoOpTimerNode':
        """Return a new no-op child timer."""
        return NoOpTimerNode(name, self.manager)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def elapsed(self) -> float:
        return 0.0
        
    def get_average_ms(self) -> float:
        """Return 0.0 for disabled timers."""
        return 0.0
    
    def get_average_exclusive_ms(self) -> float:
        """Return 0.0 for disabled timers."""
        return 0.0
    
    def get_total_ms(self) -> float:
        """Return 0.0 for disabled timers."""
        return 0.0
    
    def get_exclusive_total_ms(self) -> float:
        """Return 0.0 for disabled timers."""
        return 0.0
    
    def reset(self):
        """No-op for disabled timers."""
        pass
    
    def get_depth(self) -> int:
        """Return 0 for disabled timers."""
        return 0
    
    def get_path(self) -> str:
        """Return name for disabled timers."""
        return self.name
    
    def collect_all_nodes(self) -> List['NoOpTimerNode']:
        """Return empty list for disabled timers."""
        return []
    
    @property
    def count(self) -> int:
        """Return 0 for disabled timers."""
        return 0


class TimerNode:
    """A hierarchical timer node that tracks timing statistics with parent-child relationships."""
    
    def __init__(self, name: str, manager: 'HierarchicalTimerManager', enabled: bool = True):
        self.name = name
        self.manager = manager
        self.enabled = enabled
        self.start_time = None
        
        # Timing statistics
        self.total_time = 0.0  # Total inclusive time
        self.exclusive_time = 0.0  # Time excluding children
        self.count = 0
        
        self._current_elapsed = 0.0
        
        # Hierarchy
        self.parent: Optional['TimerNode'] = None
        self.children: Dict[str, 'TimerNode'] = {}
        
        # Track children execution time during this node's execution
        self._children_time_in_current_execution = 0.0
        
    def add_child(self, child: 'TimerNode') -> 'TimerNode':
        """Add a child node and set parent relationship."""
        child.parent = self
        self.children[child.name] = child
        return child
        
    def get_or_create_child(self, name: str):
        """Get existing child or create a new one."""
        # If manager is disabled, return a no-op timer
        if not self.manager.enabled:
            return NoOpTimerNode(name, self.manager)
            
        if name not in self.children:
            child = TimerNode(name, self.manager, self.enabled)
            self.add_child(child)
        return self.children[name]
        
    def __enter__(self):
        # Register with manager that we're entering this timer
        self.manager._enter_timer_node(self)
        
        if self.enabled:
            torch.cuda.synchronize()  # Synchronize GPU operations
            self.start_time = time.perf_counter()
            self._children_time_in_current_execution = 0.0
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.start_time is not None:
            torch.cuda.synchronize()  # Synchronize GPU operations
            elapsed = time.perf_counter() - self.start_time
            exclusive_elapsed = elapsed - self._children_time_in_current_execution
            
            # Store current execution results
            self._current_elapsed = elapsed
            
            # Update statistics
            self.total_time += elapsed
            self.exclusive_time += exclusive_elapsed
            self.count += 1
            
            # Propagate timing to parent
            if self.parent is not None:
                self.parent._children_time_in_current_execution += elapsed
        
        # Unregister with manager that we're exiting this timer
        self.manager._exit_timer_node()
    
    def get_average_ms(self) -> float:
        """Get average total time in milliseconds."""
        if self.count == 0:
            return 0.0
        return (self.total_time / self.count) * 1000.0
    
    def get_average_exclusive_ms(self) -> float:
        """Get average exclusive time in milliseconds."""
        if self.count == 0:
            return 0.0
        return (self.exclusive_time / self.count) * 1000.0
    
    def get_total_ms(self) -> float:
        """Get total time in milliseconds."""
        return self.total_time * 1000.0
    
    def get_exclusive_total_ms(self) -> float:
        """Get total exclusive time in milliseconds."""
        return self.exclusive_time * 1000.0
    
    def elapsed(self) -> float:
        return self._current_elapsed
    
    def reset(self):
        """Reset the timer statistics."""
        self.total_time = 0.0
        self.exclusive_time = 0.0
        self.count = 0
        self._current_elapsed = 0.0
        self._children_time_in_current_execution = 0.0
        
        # Reset children recursively
        for child in self.children.values():
            child.reset()
    
    def get_depth(self) -> int:
        """Get the depth of this node in the tree."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def get_path(self) -> str:
        """Get the full path from root to this node."""
        path_parts = []
        current = self
        while current is not None:
            path_parts.append(current.name)
            current = current.parent
        return " → ".join(reversed(path_parts))
    
    def collect_all_nodes(self) -> List['TimerNode']:
        """Collect all nodes in the subtree rooted at this node."""
        nodes = [self]
        for child in self.children.values():
            nodes.extend(child.collect_all_nodes())
        return nodes


class HierarchicalTimerManager:
    """Manages hierarchical timers and provides tree-structured reporting functionality."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.root_nodes: Dict[str, TimerNode] = {}
        
        # Thread-local storage for tracking the current execution stack
        self._local = threading.local()
        self._local.timer_stack = []
        
    def _get_current_stack(self) -> List[TimerNode]:
        return self._local.timer_stack
    
    def get_timer(self, name: str):
        """Get or create a hierarchical timer with the given name."""
        # If timing is disabled, return a no-op timer
        if not self.enabled:
            return NoOpTimerNode(name, self)
            
        current_stack = self._get_current_stack()
        
        if not current_stack:
            # This is a root timer
            if name not in self.root_nodes:
                self.root_nodes[name] = TimerNode(name, self, self.enabled)
            return self.root_nodes[name]
        else:
            # This is a child timer
            parent = current_stack[-1]
            return parent.get_or_create_child(name)
    
    def _enter_timer_node(self, timer_node) -> None:
        """Register that we're entering a timer context."""
        # Only track enabled TimerNode instances in the stack
        if isinstance(timer_node, TimerNode) and self.enabled:
            current_stack = self._get_current_stack()
            current_stack.append(timer_node)
    
    def _exit_timer_node(self) -> None:
        """Register that we're exiting a timer context."""
        # Only pop from stack if we have items and timing is enabled
        if self.enabled:
            current_stack = self._get_current_stack()
            if current_stack:
                current_stack.pop()
    
    def print_report(self, title: str = "Hierarchical Timing Report", show_exclusive: bool = True):
        """Print a hierarchical timing report for all timers.
        
        Args:
            title: Title for the report
            show_exclusive: Whether to show exclusive timing columns
        """
        if not self.enabled:
            return
            
        print(f"\n{'='*100}")
        print(f"{title:^100}")
        print(f"{'='*100}")
        
        if show_exclusive:
            header = f"{'Timer Path':<50} {'Avg Inc(ms)':<12} {'Avg Exc(ms)':<12} {'Total Inc(ms)':<14} {'Total Exc(ms)':<14} {'Count':<8}"
            print(header)
            print(f"{'-'*100}")
        else:
            header = f"{'Timer Path':<50} {'Avg (ms)':<12} {'Total (ms)':<14} {'Count':<8}"
            print(header)
            print(f"{'-'*100}")
        
        # Process root nodes in their natural order (no sorting)
        root_nodes = list(self.root_nodes.values())
        
        total_avg_inclusive = 0.0
        total_avg_exclusive = 0.0
        total_total_inclusive = 0.0
        total_total_exclusive = 0.0
        total_count = 0
        
        # Print each tree with preserved structure
        for root in root_nodes:
            if root.count == 0:
                continue
            self._print_tree_node(root, 0, show_exclusive)
            
            # Add to totals (only root nodes)
            total_avg_inclusive += root.get_average_ms()
            total_avg_exclusive += root.get_average_exclusive_ms()
            total_total_inclusive += root.get_total_ms()
            total_total_exclusive += root.get_exclusive_total_ms()
            total_count += root.count
        
        print(f"{'-'*100}")
        if show_exclusive:
            print(f"{'ROOT TOTALS':<50} {total_avg_inclusive:<12.3f} {total_avg_exclusive:<12.3f} {total_total_inclusive:<14.3f} {total_total_exclusive:<14.3f} {total_count:<8}")
        else:
            print(f"{'ROOT TOTALS':<50} {total_avg_inclusive:<12.3f} {total_total_inclusive:<14.3f} {total_count:<8}")
        print(f"{'='*100}\n")
    
    def _print_tree_node(self, node: 'TimerNode', depth: int, show_exclusive: bool):
        """Print a timer node and its children recursively with proper indentation."""
        indent = "  " * depth
        display_name = f"{indent}{node.name}"
        
        avg_inc_ms = node.get_average_ms()
        avg_exc_ms = node.get_average_exclusive_ms()
        total_inc_ms = node.get_total_ms()
        total_exc_ms = node.get_exclusive_total_ms()
        count = node.count
        
        if show_exclusive:
            print(f"{display_name:<50} {avg_inc_ms:>12.3f} {avg_exc_ms:>12.3f} {total_inc_ms:>14.3f} {total_exc_ms:>14.3f} {count:>8}")
        else:
            print(f"{display_name:<50} {avg_inc_ms:>12.3f} {total_inc_ms:>14.3f} {count:>8}")

        # Print children recursively
        for child in node.children.values():
            if child.count > 0:
                self._print_tree_node(child, depth + 1, show_exclusive)
    
    def reset_all(self):
        """Reset all timers and clear the entire tree structure."""
        self.root_nodes.clear()
        
        # Clear any thread-local timer stacks to avoid stale references
        self._local.timer_stack.clear()
    
    def set_enabled(self, enabled: bool):
        """Enable or disable all timers."""
        old_enabled = self.enabled
        self.enabled = enabled
        
        # If we're changing state, clear existing timers to avoid inconsistencies
        if old_enabled != enabled:
            self.reset_all()
            
        # Update existing nodes
        for root in self.root_nodes.values():
            self._set_node_enabled(root, enabled)
    
    def _set_node_enabled(self, node: TimerNode, enabled: bool):
        """Recursively set enabled state for a node and its children."""
        node.enabled = enabled
        for child in node.children.values():
            self._set_node_enabled(child, enabled)


# Global hierarchical timer manager instance for convenience
global_timer_manager = HierarchicalTimerManager(enabled=False)


def get_timer(name: str):
    """Convenience function to get a hierarchical timer from the global manager."""
    return global_timer_manager.get_timer(name)


def print_timing_report(title: str = "Hierarchical Timing Report", show_exclusive: bool = True):
    """Convenience function to print hierarchical timing report from global manager."""
    global_timer_manager.print_report(title, show_exclusive)


def reset_timer():
    """Convenience function to reset all timing statistics and clear the timer tree."""
    global_timer_manager.reset_all()


def set_timing_enabled(enabled: bool):
    """Convenience function to enable/disable timing."""
    global_timer_manager.set_enabled(enabled)
    print(f"[Timing] Hierarchical timing {'enabled' if enabled else 'disabled'}")
