"""
Agent profiler utilities for parsing flamegraph folded files.
"""

from typing import List, Tuple, Dict


def get_leaves_with_counts(folded_file_path: str) -> List[Tuple[str, int]]:
    """
    Parse a flamegraph folded file and extract all leaf functions with their sample counts.
    
    The folded file format is: func1;func2;func3;...;funcN <SAMPLE_COUNT>
    where funcN is the leaf (on-CPU sampled function) and the preceding functions
    are part of the call stack leading to it.
    
    Args:
        folded_file_path: Path to the .folded file
        
    Returns:
        A list of tuples (leaf_function_name, sample_count)
    """
    leaves = []
    
    with open(folded_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by last space to separate stack trace from count
            last_space_idx = line.rfind(' ')
            if last_space_idx == -1:
                continue
            
            stack_trace = line[:last_space_idx]
            count_str = line[last_space_idx + 1:]
            
            try:
                sample_count = int(count_str)
            except ValueError:
                continue
            
            # The leaf function is the last one in the semicolon-separated stack
            if ';' in stack_trace:
                leaf_func = stack_trace.rsplit(';', 1)[-1]
            else:
                leaf_func = stack_trace
            
            leaves.append((leaf_func, sample_count))
    
    return leaves


def get_aggregated_leaves(folded_file_path: str) -> Dict[str, int]:
    """
    Parse a flamegraph folded file and return leaf functions aggregated by total sample count.
    
    Since the same leaf function can appear in multiple stack traces, this function
    aggregates the counts across all occurrences.
    
    Args:
        folded_file_path: Path to the .folded file
        
    Returns:
        A dictionary mapping leaf function names to their total sample count
    """
    aggregated: Dict[str, int] = {}
    
    for leaf_func, sample_count in get_leaves_with_counts(folded_file_path):
        aggregated[leaf_func] = aggregated.get(leaf_func, 0) + sample_count
    
    return aggregated


def get_top_leaves(folded_file_path: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Get the top N leaf functions by sample count.
    
    Args:
        folded_file_path: Path to the .folded file
        top_n: Number of top functions to return (default: 10)
        
    Returns:
        A sorted list of tuples (leaf_function_name, total_sample_count),
        ordered by sample count descending
    """
    aggregated = get_aggregated_leaves(folded_file_path)
    sorted_leaves = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    return sorted_leaves[:top_n]


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python agent_profiler.py <folded_file_path> [top_n]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"Top {top_n} leaf functions by sample count:")
    print("-" * 60)
    
    for func, count in get_top_leaves(file_path, top_n):
        print(f"{count:>15,}  {func}")
