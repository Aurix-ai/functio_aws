"""
Agent profiler utilities for parsing flamegraph folded files.
"""

from __future__ import annotations

import json
import logging
from mimetypes import init
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from logging_setup import configure_program_logging

from function_lookup import (
    find_function_definition,
    parse_short_function_name,
    function_lookup
)
from prompt_constructor import PromptConstructor
from llm import AnthropicClaudeClient

@dataclass(frozen=True)
class FlamegraphContext:
    """
    Small convenience wrapper around a flamegraph folded file.

    Goals:
    - Avoid passing folded file path everywhere
    - Cache expensive parsing/aggregation so multiple consumers reuse it
    - Keep existing free functions usable (this class composes them)
    """

    folded_file_path: str
    demangle: bool = True
    executable: Optional[str] = None

    @cached_property
    def aggregated_leaves(self) -> Dict[str, Tuple[int, str]]:
        """Cached aggregated leaf counts keyed by demangled leaf name."""
        return get_aggregated_leaves(self.folded_file_path, demangle=self.demangle)

    def leaves_with_counts(self) -> List[Tuple[str, int, str]]:
        """Raw leaves with counts (not cached)."""
        return get_leaves_with_counts(self.folded_file_path, demangle=self.demangle)

    def top_leaves(self, top_n: int = 10) -> List[Tuple[str, int, str]]:
        """Top N leaves by total sample count."""
        sorted_leaves = sorted(
            ((name, count, mangled) for name, (count, mangled) in self.aggregated_leaves.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_leaves[:top_n]

    def top_leaves_with_locations(self, top_n: int = 10) -> List[Tuple[str, int, Optional[str]]]:
        """
        Top N leaves with source file locations resolved via nm/addr2line.

        Requires `self.executable` to be set.
        """
        if not self.executable:
            raise ValueError("FlamegraphContext.executable is required for location lookups")

        results: List[Tuple[str, int, Optional[str]]] = []
        for func_demangled, count, func_mangled in self.top_leaves(top_n=top_n):
            location = get_source_location(func_mangled, self.executable)
            results.append((func_demangled, count, location))
        return results

    def get_caller(self, func_name: str, index: int = 0) -> Optional[Tuple[str, int]]:
        """
        Return the most significant caller of `func_name` at a specific position.

        This walks one step up the call stack by finding the caller with
        the highest aggregated sample count across all stack traces where
        `func_name` appears at the given index from the leaf.

        Args:
            func_name: The function name to find the top caller for
            index: Position from leaf (0 = leaf, 1 = caller of leaf, etc.)

        Returns:
            A tuple (caller_name, sample_count), or None if no caller found
        """
        return get_next_function(self.folded_file_path, func_name, index=index, demangle=self.demangle)


@dataclass
class DepthResult:
    """
    Result from a single LLM call at a specific depth in the call stack analysis.
    
    Stores the outcome of analyzing functions at one level of the flamegraph,
    including any optimization findings and the LLM's reasoning process.
    """
    depth: int
    optimization_found: bool
    message: str
    functions_at_depth: List[str]
    scratchpad: str  # Extracted from <thinking> block


@dataclass
class FunctionMemory:
    """
    Memory structure for tracking analysis of a single leaf function across depths.
    
    Accumulates results from each depth level as the LLM walks up the call stack,
    providing context for subsequent analysis iterations.
    """
    function: str
    samples: int
    location: Optional[str]
    depth_results: List[DepthResult]


def demangle_with_tool(symbol: str, tool: str) -> str | None:
    """Try to demangle a symbol using a specific demangling tool."""
    try:
        result = subprocess.run([tool, symbol], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip() != symbol:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None

def demangle_symbol(symbol: str) -> str:
    """Try to demangle a symbol using available tools in order of preference."""
    # Try llvm-cxxfilt first (better at modern C++ symbols)
    demangled = demangle_with_tool(symbol, 'llvm-cxxfilt')
    if demangled:
        return demangled
    
    # Fall back to c++filt
    demangled = demangle_with_tool(symbol, 'c++filt')
    if demangled:
        return demangled
    
    # Return original if no demangling succeeded
    return symbol

def is_filtered_function(func_name):
    """Check if a function should be filtered out (Linux kernel, mangled names, std::)."""
    import re
    
    # Filter Linux kernel functions
    kernel_patterns = [
        r'^(sys_|__sys_|do_sys_)',
        r'^(kernel_|__kernel_)',
        r'^(sched_|__sched_)',
        r'^(mm_|__mm_)',
        r'^(vfs_|__vfs_)',
        r'^(security_|__security_)',
        r'^(arch_|__arch_)',
        r'^(irq_|__irq_)',
        r'^(rcu_|__rcu_)',
        r'^(trace_|__trace_)',
        r'^(perf_|__perf_)',
        r'^(kthread_|__kthread_)',
        r'^(mutex_|__mutex_)',
        r'^(spin_|__spin_)',
        r'^(raw_spin_)',
        r'^(kmem_cache_)',
        r'^(page_fault)',
        r'^(handle_)',
        r'^(native_)',
        r'^\[kernel\.',
        r'kernel\.kallsyms',
        r'^\[[^\]]*\.ko\]'
    ]
    for pattern in kernel_patterns:
        if re.search(pattern, func_name, re.IGNORECASE):
            return True
    return False
# =============================================================================
# Source Location Lookup
# =============================================================================

def fallback_function_definition_finder(
    name: str, 
    executable: str, 
    qualifier: Optional[str] = None,
    *,
    demangle_nm: bool = True,
) -> Optional[str]:
    """
    Find the source file location of a function using nm and addr2line.
    
    Uses nm to find the function's address in the executable, then uses
    addr2line to resolve that address to a source file location.
    
    Args:
        name: The function name to search for
        executable: Path to the executable binary
        qualifier: Optional namespace/class qualifier (e.g., "DB::")
        
    Returns:
        The source file path where the function is defined, or None if not found
    """
    SYM = f"{qualifier}{name}" if qualifier else name
    
    # Use nm to find the function symbol and grep for it.
    # When searching for a mangled symbol (e.g. _Z...), we must NOT demangle nm output,
    # otherwise the symbol text won't match.
    nm_flag = "-anC" if demangle_nm else "-an"
    demangling_command = f"nm {nm_flag} '{executable}' | rg -F '{SYM}'"
    result = subprocess.run(
        demangling_command,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0 or not result.stdout.strip():
        return None
    
    # Extract hex address from nm output (first column)
    try:
        address = result.stdout.split()[0]
    except IndexError:
        return None
    
    # Use addr2line to get source file location
    addr2line_command = f"addr2line -Cfipe '{executable}' 0x{address}"
    result = subprocess.run(
        addr2line_command,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode != 0 or not result.stdout.strip():
        return None
    
    # Parse addr2line output: "function_name at /path/to/file.cpp:123"
    try:
        if " at " in result.stdout:
            location_part = result.stdout.split(" at ")[1]
            func_location = location_part.split(":")[0].strip()
            # Filter out unknown locations
            if func_location and func_location != "??" and func_location != "":
                return func_location
    except (IndexError, ValueError):
        pass
    
    return None


def get_source_location(func_name: str, executable: str) -> Optional[str]:
    """
    Get the source file location for a function name.
    
    Tries various parsing strategies to extract a searchable function name
    from complex C++ signatures.
    
    Args:
        func_name: The function name. Prefer passing the mangled name when available.
        executable: Path to the executable binary
        
    Returns:
        The source file path, or None if not found
    """
    # Try direct lookup first.
    # If this looks mangled, search using non-demangled nm output.
    location = fallback_function_definition_finder(
        func_name,
        executable,
        demangle_nm=not func_name.startswith("_Z"),
    )
    if location:
        return location
    
    # Try extracting simpler names from complex C++ signatures
    search_names = []
    
    # Extract class::method patterns
    if '::' in func_name:
        # Get the last component (method name)
        parts = func_name.split('::')
        # Filter out anonymous namespace
        parts = [p for p in parts if p != '(anonymous namespace)' and p.strip()]
        if parts:
            # Try last two components (Class::method)
            if len(parts) >= 2:
                search_names.append(f"{parts[-2]}::{parts[-1].split('(')[0].split('<')[0]}")
            # Try just the method name
            method = parts[-1].split('(')[0].split('<')[0]
            if method:
                search_names.append(method)
    
    # Try without template parameters
    if '<' in func_name:
        base_name = func_name.split('<')[0]
        if '::' in base_name:
            search_names.append(base_name.split('::')[-1])
        else:
            search_names.append(base_name)
    
    # Try without function parameters
    if '(' in func_name:
        base_name = func_name.split('(')[0]
        if '::' in base_name:
            search_names.append(base_name.split('::')[-1])
        else:
            search_names.append(base_name)
    
    # Try each candidate
    for candidate in search_names:
        candidate = candidate.strip()
        if candidate and len(candidate) > 2:
            location = fallback_function_definition_finder(candidate, executable)
            if location:
                return location
    
    return None


def get_source_code(file_path: str, context_lines: int = 0) -> Optional[str]:
    """
    Read and return the source code from a file.
    
    Args:
        file_path: Path to the source file
        context_lines: Not used currently, reserved for future line-based extraction
        
    Returns:
        The file contents as a string, or None if the file cannot be read
    """
    if not file_path:
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except (IOError, OSError, FileNotFoundError):
        return None


# =============================================================================
# Leaf Function Extraction
# =============================================================================

def get_leaves_with_counts(folded_file_path: str, demangle: bool = True) -> List[Tuple[str, int, str]]:
    """
    Parse a flamegraph folded file and extract all leaf functions with their sample counts.
    
    The folded file format is: func1;func2;func3;...;funcN <SAMPLE_COUNT>
    where funcN is the leaf (on-CPU sampled function) and the preceding functions
    are part of the call stack leading to it.
    
    Args:
        folded_file_path: Path to the .folded file
        demangle: Whether to demangle C++ symbols (those starting with _Z)
        
    Returns:
        A list of tuples (leaf_function_name, sample_count, leaf_function_name_mangled)
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
            
            # Preserve mangled name for addr2line/nm lookups; optionally demangle for display/source parsing.
            leaf_func_mangled = leaf_func
            leaf_func_display = leaf_func

            if demangle and leaf_func_mangled.startswith('_Z'):
                leaf_func_display = demangle_symbol(leaf_func_mangled)
            
            leaves.append((leaf_func_display, sample_count, leaf_func_mangled))
    
    return leaves


def get_aggregated_leaves(folded_file_path: str, demangle: bool = True) -> Dict[str, Tuple[int, str]]:
    """
    Parse a flamegraph folded file and return leaf functions aggregated by total sample count.
    
    Since the same leaf function can appear in multiple stack traces, this function
    aggregates the counts across all occurrences.
    
    Args:
        folded_file_path: Path to the .folded file
        demangle: Whether to demangle C++ symbols (those starting with _Z)
        
    Returns:
        A dictionary mapping leaf function names to their total sample count
    """
    aggregated: Dict[str, Tuple[int, str]] = {}
    
    for leaf_func, sample_count, leaf_func_mangled in get_leaves_with_counts(folded_file_path, demangle=demangle):
        if leaf_func in aggregated:
            prev_count, prev_mangled = aggregated[leaf_func]
            aggregated[leaf_func] = (prev_count + sample_count, prev_mangled)
        else:
            aggregated[leaf_func] = (sample_count, leaf_func_mangled)
    
    return aggregated


def get_top_leaves(folded_file_path: str, top_n: int = 10, demangle: bool = True) -> List[Tuple[str, int, str]]:
    """
    Get the top N leaf functions by sample count.
    
    Args:
        folded_file_path: Path to the .folded file
        top_n: Number of top functions to return (default: 10)
        demangle: Whether to demangle C++ symbols (those starting with _Z)
        
    Returns:
        A sorted list of tuples (leaf_function_name, total_sample_count, leaf_function_name_mangled),
        ordered by sample count descending
    """
    aggregated = get_aggregated_leaves(folded_file_path, demangle=demangle)
    sorted_leaves = sorted(
        ((name, count, mangled) for name, (count, mangled) in aggregated.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return sorted_leaves[:top_n]


def get_top_leaves_with_locations(
    folded_file_path: str,
    executable: str,
    top_n: int = 10,
    demangle: bool = True
) -> List[Tuple[str, int, Optional[str]]]:
    """
    Get the top N leaf functions by sample count, with source file locations.
    
    Uses nm and addr2line to resolve function names to source file paths.
    
    Args:
        folded_file_path: Path to the .folded file
        executable: Path to the executable binary for source location lookup
        top_n: Number of top functions to return (default: 10)
        demangle: Whether to demangle C++ symbols (those starting with _Z)
        
    Returns:
        A sorted list of tuples (leaf_function_name, total_sample_count, source_location),
        ordered by sample count descending. source_location may be None if not found.
    """
    top_leaves = get_top_leaves(folded_file_path, top_n, demangle=demangle)
    
    results: List[Tuple[str, int, Optional[str]]] = []
    for func_demangled, count, func_mangled in top_leaves:
        # Resolve location using the mangled name only.
        location = get_source_location(func_mangled, executable)
        # Keep the demangled name for downstream source parsing / function lookup.
        results.append((func_demangled, count, location))
    
    return results


# =============================================================================
# Caller Lookup (walk up the call stack)
# =============================================================================

def get_callers(
    folded_file_path: str,
    func_name: str,
    index: int = 0,
    demangle: bool = True,
) -> Dict[str, int]:
    """
    Parse the folded file and return all direct callers of `func_name`
    at a specific position from the leaf, with their aggregated sample counts.

    The folded file format is: func1;func2;func3;...;funcN <SAMPLE_COUNT>
    
    Index represents position from the leaf:
    - index 0 = leaf (last function in trace)
    - index 1 = immediate caller of leaf
    - index N = function at position len(stack) - 1 - N

    Args:
        folded_file_path: Path to the .folded file
        func_name: The function name to find callers for
        index: Position from leaf (0 = leaf, 1 = caller of leaf, etc.)
        demangle: Whether to demangle C++ symbols (those starting with _Z)

    Returns:
        A dictionary mapping caller function names to their total sample count
    """
    callers: Dict[str, int] = {}

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

            # Split the stack into individual functions
            stack_funcs = stack_trace.split(';')

            # Position from end: index 0 = last element (leaf)
            pos = len(stack_funcs) - 1 - index
            if pos < 0 or pos >= len(stack_funcs):
                continue  # index out of range for this trace

            # Get the function at the expected position
            func_at_pos = stack_funcs[pos]
            func_display = func_at_pos
            if demangle and func_at_pos.startswith('_Z'):
                func_display = demangle_symbol(func_at_pos)

            # Check if this matches func_name
            if func_display == func_name or func_at_pos == func_name:
                # Found the function at the expected position; caller is the previous one
                if pos > 0:
                    caller = stack_funcs[pos - 1]
                    caller_display = caller
                    if demangle and caller.startswith('_Z'):
                        caller_display = demangle_symbol(caller)

                    callers[caller_display] = callers.get(caller_display, 0) + sample_count

    return callers


def get_next_function(
    folded_file_path: str,
    func_name: str,
    index: int = 0,
    demangle: bool = True,
) -> Optional[Tuple[str, int]]:
    """
    Return the caller of `func_name` at a specific position from the leaf,
    with the highest sample count.

    This is useful for walking up the call stack in a flamegraph to find
    the most significant caller of a hot function.

    Args:
        folded_file_path: Path to the .folded file
        func_name: The function name to find the top caller for
        index: Position from leaf (0 = leaf, 1 = caller of leaf, etc.)
        demangle: Whether to demangle C++ symbols (those starting with _Z)

    Returns:
        A tuple (caller_name, sample_count) for the caller with the highest
        sample count, or None if func_name has no callers (e.g., it's at the
        root of every trace).
    """
    callers = get_callers(folded_file_path, func_name, index=index, demangle=demangle)
    if not callers:
        return None
    top_caller = max(callers.items(), key=lambda x: x[1])
    return top_caller

def llm_loop(
    ctx: FlamegraphContext,
    initial_leaf_function: str,
    leaf_function_location: str,
    query: str,
    max_depth: int = 3,
):
    """
    Walk up the call stack to a specified depth, calling LLM at each level.

    Even if an optimization is found, continues walking up to max_depth levels
    to gather more context and potentially find better optimizations.

    Args:
        ctx: FlamegraphContext for caller lookups
        initial_leaf_function: The starting leaf function name
        leaf_function_location: Source location of the initial function
        query: The SQL query being profiled
        max_depth: Maximum depth to walk up the call stack (default: 3)

    Returns:
        Tuple of (functions_analyzed, list of results from each depth level)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"llm_loop started for function: {initial_leaf_function}")
    logger.info(f"Initial location: {leaf_function_location}")
    logger.info(f"Max depth: {max_depth}")
    
    # Create prompt builder once for all iterations
    prompt_builder = PromptConstructor.create()
    llm_client = AnthropicClaudeClient()
    
    # Track both function names and their locations
    functions_to_check_by_agent = [initial_leaf_function]
    function_locations = [leaf_function_location]
    current_index = 0  # starts at leaf (index 0)
    
    # Collect results from each depth level
    all_results: List[Tuple[int, bool, str, List[str]]] = []  # (depth, status, message, functions)
    
    if is_filtered_function(initial_leaf_function):
        logger.info(f"Function {initial_leaf_function} is one of the kernel functions therefore we should not continue the llm loop:")
        return functions_to_check_by_agent, [(0, False, "Kernel Function, aborting", functions_to_check_by_agent.copy())]

    for depth in range(max_depth):
        logger.info(f"llm_loop depth {depth + 1}/{max_depth}: analyzing {len(functions_to_check_by_agent)} function(s)")
        logger.debug(f"Functions in trace: {' -> '.join(functions_to_check_by_agent)}")
        
        llm_call_result = llm_call(
            functions_to_check_by_agent,
            function_locations,
            ctx.executable,
            prompt_builder,
            llm_client,
            query,
        )
        
        status = llm_call_result[0]
        message = llm_call_result[1]
        
        # Record result for this depth
        all_results.append((depth, status, message, functions_to_check_by_agent.copy()))
        
        if status:
            logger.info(f"Optimization found at depth {depth + 1}!")
            logger.info(f"Summary: {message}")
        else:
            logger.info(f"No optimization found at depth {depth + 1}")
        
        # If we haven't reached max_depth, walk up to the next caller
        if depth < max_depth - 1:
            next_result = ctx.get_caller(functions_to_check_by_agent[-1], index=current_index)
            if next_result:
                next_function_name, _sample_count = next_result
                
                # Check if the next function is a kernel function
                if is_filtered_function(next_function_name):
                    logger.info(f"Next caller {next_function_name} is a kernel function - stopping at depth {depth + 1}")
                    break
                
                logger.info(f"Walking up to caller: {next_function_name} (samples: {_sample_count})")
                functions_to_check_by_agent.append(next_function_name)
                
                # Look up the location of the new function
                next_location = None
                if ctx.executable:
                    next_location = get_source_location(next_function_name, ctx.executable)
                    logger.info(f"Caller location: {next_location if next_location else '<unknown>'}")
                function_locations.append(next_location)
                
                current_index += 1  # walked up one level
            else:
                logger.info(f"No more callers found - stopping at depth {depth + 1}")
                break

    if not ctx.executable:
        logger.warning("No executable provided - returning None")
        return None

    logger.info(f"llm_loop completed. Total functions analyzed: {len(functions_to_check_by_agent)}")
    logger.info(f"Final function trace: {' -> '.join(functions_to_check_by_agent)}")
    logger.info(f"Results collected from {len(all_results)} depth level(s)")
    
    # Log summary of all results
    optimizations_found = sum(1 for r in all_results if r[1])
    logger.info(f"Optimizations found: {optimizations_found}/{len(all_results)} levels")
    
    return functions_to_check_by_agent, all_results


def llm_call(
    function_list: List[str],
    location_list: List[Optional[str]],
    executable: Optional[str],
    prompt_builder: PromptConstructor,
    llm_client: AnthropicClaudeClient,
    query: str,
) -> Tuple[bool, str, str]:
    """
    Make an LLM call to evaluate the set of functions.

    Args:
        function_list: List of function names to check
        location_list: List of source file locations for each function (may contain None)
        executable: Path to the executable binary for function lookup
        prompt_builder: PromptConstructor instance for building prompts
        query: The SQL query being profiled

    Returns:
        A tuple of (status, message, scratchpad) where:
        - status: True if an optimization was found, False otherwise
        - message: The optimization description or reason for no optimization
        - scratchpad: The content of the <thinking> block from the LLM response
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"llm_call: Analyzing {len(function_list)} function(s)")
    logger.debug(f"Function list: {function_list}")
    
    # Collect source code for all functions
    all_source_code_parts: List[str] = []
    functions_found = 0
    functions_not_found = 0

    for func_name, location in zip(function_list, location_list):
        if location is None or executable is None:
            # Can't look up source for this function
            logger.debug(f"Function '{func_name}': location unknown")
            all_source_code_parts.append(f"// Function: {func_name}\n// Location: unknown\n")
            functions_not_found += 1
            continue

        # Parse the short function name for lookup
        short_name = parse_short_function_name(func_name)

        # Read the source file
        try:
            with open(location, 'r', encoding='utf-8', errors='replace') as f:
                source_code = f.read()
        except (IOError, OSError, FileNotFoundError):
            logger.warning(f"Function '{func_name}': could not read source file at {location}")
            all_source_code_parts.append(f"// Function: {func_name}\n// Location: {location}\n// Could not read source file\n")
            functions_not_found += 1
            continue

        # Find the function definition in the source
        result = find_function_definition(source_code, short_name)

        if result:
            start, brace_open, brace_close = result
            # Extract the function source code (from signature start to closing brace)
            func_source = source_code[start:brace_close + 1]
            all_source_code_parts.append(
                f"// Function: {func_name}\n"
                f"// Location: {location}\n"
                f"{func_source}\n"
            )
            logger.debug(f"Function '{func_name}': found definition ({brace_close - start + 1} chars)")
            functions_found += 1
        else:
            logger.warning(f"Function '{func_name}': could not find definition in {location}")
            all_source_code_parts.append(
                f"// Function: {func_name}\n"
                f"// Location: {location}\n"
                f"// Could not find function definition\n"
            )
            functions_not_found += 1

    logger.info(f"Source extraction: {functions_found} found, {functions_not_found} not found")

    # If no source code was found for any function, skip the LLM call
    if functions_found == 0:
        logger.warning("No source code found for any function - skipping LLM call")
        return False, "No source code found for any function - LLM call skipped", ""

    # Combine all source code into one string
    combined_source_code = "\n" + "=" * 80 + "\n".join(all_source_code_parts)
    # Build function call trace string from function_list
    function_trace = " -> ".join(function_list)

    # Construct the prompt based on number of functions
    template_name = "single_function" if len(function_list) == 1 else "multiple_functions"
    logger.info(f"Using template: {template_name}")
    
    if len(function_list) == 1:
        prompt = prompt_builder.construct(
            template_name="single_function",
            context={
                "QUERY": query,
                "SOURCE_CODE": combined_source_code,
            }
        )
    else:
        prompt = prompt_builder.construct(
            template_name="multiple_functions",
            context={
                "QUERY": query,
                "FUNCTION_TRACE": function_trace,
                "SOURCE_CODE": combined_source_code,
            }
        )
    
    logger.info(f"Prompt constructed ({len(prompt)} chars). Sending to LLM...")
    response = llm_client.query(prompt, "flamegraph").text #full response, string
    logger.info(f"LLM response received ({len(response)} chars)")
    logger.debug(f"LLM response:\n{response[:500]}..." if len(response) > 500 else f"LLM response:\n{response}")
    print(response)
    
    # Parse <thinking> block for scratchpad
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    scratchpad = thinking_match.group(1).strip() if thinking_match else ""
    if scratchpad:
        logger.debug(f"Extracted scratchpad ({len(scratchpad)} chars)")
    
    # Parse response for <optimization_available> or <no_optimization_available>
    optimization_match = re.search(
        r'<optimization_available>(.*?)</optimization_available>',
        response,
        re.DOTALL
    )
    no_optimization_match = re.search(
        r'<no_optimization_available>(.*?)</no_optimization_available>',
        response,
        re.DOTALL
    )
    
    if optimization_match:
        status = True
        message = optimization_match.group(1).strip()
        logger.info("LLM result: OPTIMIZATION AVAILABLE")
        logger.info(f"Optimization summary: {message[:200]}..." if len(message) > 200 else f"Optimization summary: {message}")
    elif no_optimization_match:
        status = False
        message = no_optimization_match.group(1).strip()
        logger.info("LLM result: NO OPTIMIZATION AVAILABLE")
        logger.debug(f"Message: {message}")
    else:
        # Fallback if neither tag found
        status = False
        message = "Could not parse LLM response"
        logger.warning("LLM result: COULD NOT PARSE RESPONSE")
        logger.warning(f"Response did not contain expected tags. First 500 chars: {response[:500]}")
    
    return status, message, scratchpad
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze flamegraph folded files')
    parser.add_argument('folded_file', help='Path to the .folded file')
    parser.add_argument('--top', '-n', type=int, default=10, help='Number of top functions to show')
    parser.add_argument('--executable', '-e', help='Path to executable for source location lookup')
    parser.add_argument('--no-demangle', action='store_true', help='Disable symbol demangling')
    parser.add_argument('--query', '-q', default='SELECT * FROM NUMBERS_MT(100000)', help='SQL query being profiled')
    parser.add_argument('--depth', '-d', type=int, default=3, help='Maximum depth to walk up the call stack (default: 3)')
    
    args = parser.parse_args()
    
    # Setup logging
    run_timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    base_run_dir = Path(f"server_logs/logs_{run_timestamp}")
    base_run_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = base_run_dir / "full_run_log.log"
    logging_handle = configure_program_logging(
        enabled=True,
        log_file=str(log_file_path),
        level=logging.INFO,
        to_console=True,
        append=True,
        capture_prints=True,
        fmt='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 60)
        logger.info(f"Agent Profiler started at {run_timestamp}")
        logger.info(f"Folded file: {args.folded_file}")
        logger.info(f"Executable: {args.executable}")
        logger.info(f"Top N: {args.top}")
        logger.info(f"Query: {args.query}")
        logger.info(f"Max depth: {args.depth}")
        logger.info("=" * 60)
        
        demangle = not args.no_demangle
        
        ctx = FlamegraphContext(
            folded_file_path=args.folded_file,
            demangle=demangle,
            executable=args.executable,
        )
        
        if args.executable:
            logger.info("Starting analysis with executable - will resolve source locations")
            results = ctx.top_leaves_with_locations(top_n=args.top)
            logger.info(f"Found {len(results)} top leaf functions")
            
            # Collect all results for JSON export
            all_run_results = []
            
            for i, (func, count, location) in enumerate(results, 1):
                logger.info("-" * 60)
                logger.info(f"[{i}/{len(results)}] Analyzing function: {func}")
                logger.info(f"    Samples: {count:,}")
                logger.info(f"    Location: {location if location else '<unknown>'}")
                
                print("=" * 120)
                print(f"[{i}] FUNCTION: {func}")
                print(f"    SAMPLES: {count:,}")
                print(f"    LOCATION: {location if location else '<unknown>'}")
                print("=" * 120)
                
                functions, all_results = llm_loop(ctx, func, location, args.query, max_depth=args.depth)
                
                logger.info(f"    LLM loop completed. Functions analyzed: {functions}")
                logger.info(f"    Results from {len(all_results)} depth level(s):")
                
                for depth, status, message, funcs_at_depth in all_results:
                    status_str = "OPTIMIZATION FOUND" if status else "No optimization"
                    logger.info(f"      Depth {depth + 1}: {status_str}")
                    logger.info(f"        Functions: {' -> '.join(funcs_at_depth)}")
                    if status:
                        logger.info(f"        Message: {message[:200]}..." if len(message) > 200 else f"        Message: {message}")
                
                # Accumulate results for JSON export
                all_run_results.append({
                    "function": func,
                    "samples": count,
                    "location": location,
                    "functions_analyzed": functions,
                    "depth_results": [
                        {
                            "depth": depth,
                            "optimization_found": status,
                            "message": message,
                            "functions_at_depth": funcs_at_depth,
                        }
                        for depth, status, message, funcs_at_depth in all_results
                    ]
                })
            
            # Save all results to JSON file
            results_file = base_run_dir / "results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": run_timestamp,
                    "folded_file": args.folded_file,
                    "executable": args.executable,
                    "query": args.query,
                    "max_depth": args.depth,
                    "top_n": args.top,
                    "results": all_run_results,
                }, f, indent=2)
            logger.info(f"Results saved to {results_file}")
                
        else:
            logger.info("No executable provided - listing top leaf functions only")
            print(f"Top {args.top} leaf functions by sample count:")
            print("-" * 60)
            
            for func, count, _func_mangled in ctx.top_leaves(top_n=args.top):
                print(f"{count:>15,}  {func}")
                logger.info(f"Leaf function: {func} ({count:,} samples)")
        
        logger.info("=" * 60)
        logger.info("Agent Profiler completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Agent Profiler failed with error: {e}", exc_info=True)
        raise
    
    finally:
        if logging_handle and hasattr(logging_handle, 'stop'):
            logging_handle.stop()

