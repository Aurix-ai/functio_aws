#!/usr/bin/env python3
"""
FlameGraph Analyzer - Unified Analysis Tool

This script generates flame graphs and analyzes them to identify performance hotspots
using the Wide Plateau Identification strategy. It can profile running processes or
launch new executables for analysis.


Requirements:
    - flame_folded_pid.sh must be executable and in the current directory
    - sudo access (required by flame_folded.sh for perf profiling)
    - perf tools installed on the system
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
import re
import shlex


def demangle_with_tool(symbol, tool):
    """Try to demangle a symbol using a specific demangling tool."""
    try:
        result = subprocess.run([tool, symbol], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip() != symbol:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def demangle_symbol(symbol):
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


def apply_enhanced_demangling(folded_path, verbose=False):
    """Apply enhanced demangling to a folded flamegraph file."""
    enhanced_path = folded_path + ".enhanced"
    demangled_count = 0
    total_lines = 0
    
    try:
        with open(folded_path, 'r', encoding='utf-8') as input_file:
            with open(enhanced_path, 'w', encoding='utf-8') as output_file:
                for line in input_file:
                    total_lines += 1
                    original_line = line
                    
                    if not line.strip():
                        output_file.write(line)
                        continue
                    
                    parts = line.rsplit(' ', 1)
                    if len(parts) != 2:
                        output_file.write(line)
                        continue
                    
                    stack_trace, count = parts
                    
                    # Split the stack trace by semicolons
                    functions = stack_trace.split(';')
                    
                    # Demangle any C++ symbols (those starting with _Z)
                    demangled_functions = []
                    line_modified = False
                    for func in functions:
                        if func.startswith('_Z'):
                            demangled = demangle_symbol(func)
                            demangled_functions.append(demangled)
                            if demangled != func:
                                line_modified = True
                        else:
                            demangled_functions.append(func)
                    
                    # Reconstruct the line
                    new_stack_trace = ';'.join(demangled_functions)
                    processed_line = f"{new_stack_trace} {count}"
                    
                    if line_modified:
                        demangled_count += 1
                        if verbose:
                            print(f"[*] Enhanced demangling on line {total_lines}")
                    
                    output_file.write(processed_line)
        
        if verbose:
            print(f"[*] Processed {total_lines} lines, enhanced demangling on {demangled_count} lines")
        
        return enhanced_path
        
    except Exception as e:
        print(f"Warning: Enhanced demangling failed: {e}", file=sys.stderr)
        return folded_path  # Return original path if demangling fails



def run_flamegraph(executable_command=None, pid=None, duration=10, output_base="flamegraph"):
    """Run flame_folded._newsh to generate flame graph folded file.
    
    Args:
        executable_command: Command to profile (mutually exclusive with pid)
        pid: Process ID to profile (mutually exclusive with executable_command)
        output_base: Base name for output files
    """
    flame_script = Path("./flame_folded_pid.sh")
    
    if not flame_script.is_file():
        raise FileNotFoundError("flame_folded_pid.sh not found in current directory")
    
    if not os.access(flame_script, os.X_OK):
        raise PermissionError("flame_folded_pid.sh is not executable")
    
    if pid and executable_command:
        print(f"[*] Running flame_folded_pid.sh for PID: {pid}")
        print(f"Executable command is {executable_command}")
        cmd_args = ["sudo", "./flame_folded_pid.sh", "--pid", str(pid), "--", str(executable_command)]
        print("successfully ran flame_folded_pid.sh")
    else:
        raise ValueError("pid or executable_command not specified")
    try:
        result = subprocess.run(cmd_args, capture_output=True, timeout=1800)
        
        # Decode with error handling for binary output
        stdout_text = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
        stderr_text = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
        
        # Check if there were errors in stderr even if exit code is 0
        if stderr_text and any(marker in stderr_text for marker in ['Exception', 'Error:', 'DB::Exception', 'error']):
            print(f"Error detected in command output:", file=sys.stderr)
            print(f"stdout: {stdout_text}", file=sys.stderr)
            print(f"stderr: {stderr_text}", file=sys.stderr)
            raise subprocess.CalledProcessError(
                result.returncode if result.returncode != 0 else 1,
                cmd_args,
                output=stdout_text,
                stderr=stderr_text
            )
        
        # Check exit code
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
            print(f"stdout: {stdout_text}", file=sys.stderr)
            print(f"stderr: {stderr_text}", file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd_args, output=stdout_text, stderr=stderr_text)
        
        print(f"[✓] FlameGraph folded file generated successfully")
        return f"flamegraph.folded"
    
    except subprocess.TimeoutExpired as e:
        stdout_output = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
        stderr_output = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
        print(f"Error: FlameGraph generation timed out after 600 seconds", file=sys.stderr)
        if stdout_output:
            print(f"stdout: {stdout_output}", file=sys.stderr)
        if stderr_output:
            print(f"stderr: {stderr_output}", file=sys.stderr)
        raise
        
    except subprocess.CalledProcessError as e:
        # Error details already printed above, just raise
        raise


def read_folded_file(folded_path):
    """Read the folded flame graph text file."""
    folded_file = Path(folded_path)
    if not folded_file.exists():
        raise FileNotFoundError(f"Folded file not found: {folded_path}")
    
    with open(folded_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content


def extract_source_locations(perf_data_path, full_paths=False):
    """Extract source line information using perf report.
    
    Args:
        perf_data_path: Path to perf.data file
        full_paths: If True, use --full-source-path to get absolute paths instead of relative ones
    """
    try:
        # Run perf report to get detailed symbol and source line information
        cmd = [
            'sudo', 'perf', 'report', '--stdio', '-i', perf_data_path, 
            '--no-children', '-F', 'overhead,comm,dso,sym,srcline'
        ]
        
        # Add full source path option if requested
        if full_paths:
            cmd.append('--full-source-path')
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse the perf report output to create a function -> location mapping
        function_locations = {}
        
        lines = output.split('\n')
        for line_num, line in enumerate(lines):
            original_line = line
            line = line.strip()
            if not line or line.startswith('#') or 'Overhead' in line or '-----' in line:
                continue
            
            # Skip call graph lines that start with |, ---, or spaces followed by ---
            if line.startswith(('|', '---')) or (line.startswith(' ') and '---' in line):
                continue
                
            # Parse lines using regex to handle complex C++ function names
            # Format: "  1.23%  program  program  [.] function_name  /path/to/file.c:123"
            import re
            
            # Look for the pattern: percentage, command, shared object, [.], symbol, source location
            # We need to be careful with C++ template names that contain spaces, commas, etc.
            match = re.match(r'^\s*(\d+\.\d+%)\s+(\S+)\s+(\S+)\s+(\[[^\]]+\])\s+(.*?)\s+([^/\s]*[./][^/\s]*:\d+)\s*$', line)
            
            if match:
                percentage = match.group(1)
                command = match.group(2)
                shared_object = match.group(3)
                bracket = match.group(4)
                symbol_and_maybe_more = match.group(5)
                srcline = match.group(6)
                
                # Extract just the symbol part (everything before the source location)
                # The symbol might be the entire remaining text before source location
                symbol = symbol_and_maybe_more.strip()
                
                if symbol and srcline:
                    # Store multiple variations of the function name for better matching
                    variations = []
                    
                    # 1. Original symbol
                    variations.append(symbol)
                    
                    # 2. Clean symbol (remove addresses, etc.)
                    clean_symbol = clean_function_name(symbol)
                    variations.append(clean_symbol)
                    
                    # 3. Base name without parameters
                    base_name = symbol.split('(')[0] if '(' in symbol else symbol
                    variations.append(base_name)
                    
                    # 4. Function name without namespace/class (for simple matching)
                    if '::' in symbol:
                        simple_name = symbol.split('::')[-1].split('(')[0]
                        variations.append(simple_name)
                    
                    # 5. Extract class::method patterns (for C++ methods)
                    class_method_match = re.search(r'([A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*)', symbol)
                    if class_method_match:
                        variations.append(class_method_match.group(1))
                    
                    # 6. Handle template instantiations - extract base template name
                    template_match = re.search(r'([A-Za-z_][A-Za-z0-9_:]*)<.*?>', symbol)
                    if template_match:
                        variations.append(template_match.group(1))
                    
                    # Store all variations, optionally keeping full paths
                    for var in variations:
                        if var and var != '' and var not in function_locations:
                            # Keep full paths if requested, otherwise convert to project-relative
                            processed_srcline = srcline
                            if not full_paths and srcline and srcline.startswith('/'):
                                processed_srcline = make_path_relative_to_project(srcline)
                            function_locations[var] = processed_srcline
                
            else:
                # Fallback to the old parsing method for lines that don't match the regex
                # This handles edge cases and malformed lines
                parts = line.split()
                if len(parts) >= 5:
                    # Find the bracket notation [.], [k], etc.
                    bracket_idx = -1
                    for i, part in enumerate(parts):
                        if part.startswith('[') and part.endswith(']') and len(part) > 2:
                            bracket_idx = i
                            break
                    
                    if bracket_idx >= 0 and bracket_idx < len(parts) - 2:
                        # Everything after bracket until we find source location
                        symbol_parts = []
                        srcline = None
                        
                        for i in range(bracket_idx + 1, len(parts)):
                            part = parts[i]
                            # Check if this looks like a source location
                            if ':' in part and ('.' in part or '/' in part) and re.search(r':\d+', part):
                                srcline = part
                                break
                            else:
                                symbol_parts.append(part)
                        
                        if symbol_parts and srcline:
                            symbol = ' '.join(symbol_parts)
                            if symbol:
                                clean_symbol = clean_function_name(symbol)
                                # Keep full paths if requested, otherwise convert to project-relative
                                processed_srcline = srcline
                                if not full_paths and srcline and srcline.startswith('/'):
                                    processed_srcline = make_path_relative_to_project(srcline)
                                function_locations[symbol] = processed_srcline
                                function_locations[clean_symbol] = processed_srcline
        
        print(f"[*] Extracted source locations for {len(function_locations)} functions")
        
        # Debug: show first few entries
        # if function_locations:
        #     print("[*] Sample source locations found:")
        #     for i, (func, loc) in enumerate(list(function_locations.items())[:5]):
        #         print(f"  {func} -> {loc}")
        # else:
        #     print("[*] No source locations found - debugging perf report output:")
        #     print(f"[*] Perf report output (first 500 chars): {output[:500]}")
        
        return function_locations
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not extract source locations from perf report: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Warning: Error extracting source locations: {e}", file=sys.stderr)
        return {}




def make_path_relative_to_project(full_path, project_hints=None):
    """Convert absolute path to project-relative path.
    
    Args:
        full_path: Full absolute path like /media/makar/Data/ClickHouse/src/Common/RadixSort.h:332
        project_hints: List of common project directory patterns to look for
    """
    if not full_path or ':' not in full_path:
        return full_path
    
    path_part, line_part = full_path.rsplit(':', 1)
    
    # Default project directory hints
    if project_hints is None:
        project_hints = ['src/', 'include/', 'lib/', 'source/', 'Sources/', 'Source/']
    
    # Try to find a project root indicator in the path
    for hint in project_hints:
        if hint in path_part:
            # Find the last occurrence of the hint and extract from there
            idx = path_part.rfind(hint)
            if idx >= 0:
                relative_path = path_part[idx:]
                return f"{relative_path}:{line_part}"
    
    # Fallback: try to find common project patterns
    import os
    path_components = path_part.split(os.sep)
    
    # Look for components that suggest project structure
    project_indicators = ['src', 'include', 'source', 'Sources', 'lib', 'libraries']
    for i, component in enumerate(path_components):
        if component.lower() in [p.lower() for p in project_indicators]:
            relative_path = os.sep.join(path_components[i:])
            return f"{relative_path}:{line_part}"
    
    # If no project structure found, return just the filename
    filename = os.path.basename(path_part)
    return f"{filename}:{line_part}"


def find_best_source_location_match(leaf_function, raw_leaf_function, source_locations):
    """Find the best matching source location for a function name."""
    if not source_locations:
        return None
    
    import re
    
    # Try exact matches first
    candidates = [leaf_function, raw_leaf_function]
    
    # Add variations
    base_name = raw_leaf_function.split('(')[0] if '(' in raw_leaf_function else raw_leaf_function
    candidates.append(base_name)
    
    # Add cleaned version
    candidates.append(clean_function_name(raw_leaf_function))
    
    # For C++ methods, try class::method patterns
    if '::' in raw_leaf_function:
        # Extract potential class::method patterns
        class_methods = re.findall(r'([A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*)', raw_leaf_function)
        candidates.extend(class_methods)
        
        # Also try just the method name
        method_name = raw_leaf_function.split('::')[-1].split('(')[0]
        candidates.append(method_name)
    
    # For template functions, try the base template name
    template_match = re.search(r'([A-Za-z_][A-Za-z0-9_:]*)<.*?>', raw_leaf_function)
    if template_match:
        candidates.append(template_match.group(1))
    
    # Try all candidates for exact matches
    for candidate in candidates:
        if candidate in source_locations:
            location = source_locations[candidate]
            # Strip line number from the end (e.g., "/path/file.h:315" -> "/path/file.h")
            if location and ':' in location:
                location = location.rsplit(':', 1)[0]
            return location
    
    # Try fuzzy matching - look for partial matches
    for candidate in candidates:
        if not candidate:
            continue
        # Look for any source location key that contains this candidate or vice versa
        for src_key, src_location in source_locations.items():
            # Skip very short matches to avoid false positives
            if len(candidate) < 5:
                continue
            
            # Check if candidate is a substring of the source key or vice versa
            if (candidate in src_key or src_key in candidate) and len(src_key) > 5:
                # Strip line number from the end (e.g., "/path/file.h:315" -> "/path/file.h")
                if src_location and ':' in src_location:
                    src_location = src_location.rsplit(':', 1)[0]
                return src_location
    
    return None


def clean_function_name(func_name):
    """Clean function name by removing addresses, template parameters, etc."""
    
    # Remove addresses like [kernel.kallsyms] or +0x123, but preserve operator[] and similar
    # Only remove brackets that look like module/library names
    if '[' in func_name:
        # Pattern to match common module/library brackets like [kernel.kallsyms], [libc.so.6], [unknown]
        # but NOT operator[] or array indexing in function signatures
        module_pattern = r'\[(?:kernel\.kallsyms|.*\.so(?:\.\d+)*|.*\.ko|unknown|vdso|.*\.a)\]'
        
        # Check if this looks like a module reference at the end or standalone
        if re.search(module_pattern, func_name):
            # Remove the module reference
            func_name = re.sub(module_pattern, '', func_name)
        else:
            # Check if it's a simple bracket pattern that's not operator[]
            # If the bracket content doesn't contain :: or () or look like an operator, remove it
            bracket_content_match = re.search(r'\[([^\]]+)\]', func_name)
            if bracket_content_match:
                bracket_content = bracket_content_match.group(1)
                # Don't remove if it looks like part of a function signature (empty brackets, or followed by parentheses)
                # Also preserve if it's right after "operator" 
                if not (func_name.count('operator') > 0 and '[]' in func_name):
                    # Only remove if it looks like an address or module, not a function signature
                    if (not '::' in bracket_content and 
                        not '(' in bracket_content and 
                        not ')' in bracket_content and
                        not bracket_content.isdigit() and  # preserve array indices
                        not 'operator' in func_name[max(0, func_name.find('[') - 20):func_name.find('[')]):  # check context
                        func_name = func_name.split('[')[0]
    
    if '+0x' in func_name:
        func_name = func_name.split('+0x')[0]
    
    # Remove leading/trailing whitespace
    func_name = func_name.strip()
    
    # If function name is empty after cleaning, use a placeholder
    if not func_name:
        func_name = "<unknown>"
    
    return func_name


def extract_file_location(func_name):
    """Extract file location from function name if available."""
    # Look for patterns like [file.so], [module.ko], or file paths
    if '[' in func_name and ']' in func_name:
        # Extract content between brackets
        start = func_name.find('[')
        end = func_name.find(']', start)
        if start != -1 and end != -1:
            location = func_name[start+1:end]
            # Return full path if it looks like a file path
            if '/' in location or location.endswith(('.so', '.ko', '.a')):
                return location
    
    # Look for patterns in the original function name that might indicate file paths
    # This is a heuristic approach for flamegraph data
    original_parts = func_name.split()
    for part in original_parts:
        if '/' in part and ('.' in part or part.startswith('/')):
            return part
    
    return None


def is_custom_function(func_name, file_location):
    """Determine if a function is custom (application-defined) vs standard library/kernel."""
    # Standard library and kernel function indicators
    standard_indicators = [
        'kernel.kallsyms', 'libc.so', 'libstdc++.so', 'libgcc.so', 'libm.so',
        'libpthread.so', 'libdl.so', 'librt.so', 'ld-linux', 'vdso',
        '[kernel.kallsyms]', '[vdso]'
    ]
    
    # Kernel function prefixes
    kernel_prefixes = [
        'sys_', '__sys_', 'do_sys_', 'kernel_', '__kernel_', 
        'sched_', '__sched_', 'mm_', '__mm_', 'vfs_', '__vfs_',
        'security_', '__security_', 'arch_', '__arch_'
    ]
    
    # Standard library function patterns
    stdlib_patterns = [
        'std::', '__gnu_cxx::', '__cxxabiv1::', '_G_',
        'malloc', 'free', 'calloc', 'realloc',
        'printf', 'scanf', 'fopen', 'fclose', 
        'pthread_', 'sem_', 'mutex_'
    ]
    
    # Check file location first
    if file_location:
        for indicator in standard_indicators:
            if indicator in file_location:
                return False
    
    # Check function name patterns
    func_lower = func_name.lower()
    
    # Check kernel function prefixes
    for prefix in kernel_prefixes:
        if func_lower.startswith(prefix):
            return False
    
    # Check standard library patterns
    for pattern in stdlib_patterns:
        if pattern in func_name:
            return False
    
    # If function name looks like a memory address or unknown, not custom
    if func_name in ['<unknown>', '0x', ''] or func_name.startswith('0x'):
        return False
    
    # Default to custom if no standard indicators found
    return True


def find_custom_parent(functions, source_locations=None):
    """Find the first custom function in the parent stack (going backwards)."""
    if source_locations is None:
        source_locations = {}
    
    # Go through functions in reverse order (from deeper to shallower in call stack)
    for func in reversed(functions):
        clean_func = clean_function_name(func.strip())
        raw_func = func.strip()
        
        # Use improved matching to find source location
        file_location = find_best_source_location_match(clean_func, raw_func, source_locations)
        
        if not file_location:
            file_location = extract_file_location(raw_func)
        
        if is_custom_function(clean_func, file_location):
            # Only return if we have a valid file location
            if file_location and file_location != 'unknown location':
                return f"{clean_func}, {file_location}"
    
    return None

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
    
    # Filter mangled C++ names (start with _Z)
    if re.match(r'^_Z', func_name):
        return True
    
    # Filter std:: functions (anywhere in the name)
    if 'std::' in func_name:
        return True
    
    # Filter other common C++ standard library patterns
    cpp_stdlib_patterns = [
        r'^__gnu_cxx::',
        r'^__cxxabiv1::',
        r'^_G_',
        r'^__libc_',
        r'^__gxx_'
    ]
    
    for pattern in cpp_stdlib_patterns:
        if re.search(pattern, func_name):
            return True
    
    return False


def is_function_in_excluded_dirs(source_location, excluded_dirs):
    """Check if a function's source location is in any of the excluded directories."""
    if not source_location or not excluded_dirs:
        return False
    
    # Extract the file path part (remove line number)
    if ':' in source_location:
        file_path = source_location.rsplit(':', 1)[0]
    else:
        file_path = source_location
    
    # Check if the file path starts with any excluded directory
    for excluded_dir in excluded_dirs:
        # Normalize paths for comparison
        excluded_dir = excluded_dir.rstrip('/\\')  # Remove trailing slashes
        
        # Check if file path starts with excluded directory
        if file_path.startswith(excluded_dir + '/') or file_path.startswith(excluded_dir + '\\') or file_path == excluded_dir:
            return True
    
    return False

    # Helper function to strip prefix path and add custom prefix
def transform_path(path, strip_prefix, add_prefix):
    """Strip prefix from path if it starts with prefix, then prepend custom prefix."""
    if not path or path == "unknown location":
        return path
    
    # Step 1: Strip the prefix if provided
    if strip_prefix:
        # Normalize prefix (remove trailing slash)
        strip_prefix = strip_prefix.rstrip('/\\')
        # Check if path starts with prefix
        if path.startswith(strip_prefix):
            # Remove prefix and return path starting with /
            path = path[len(strip_prefix):]
            # Ensure it starts with /
            if not path.startswith('/'):
                path = '/' + path
    
    # Step 2: Prepend custom prefix if provided
    if add_prefix:
        # Normalize custom prefix (remove trailing slash)
        add_prefix = add_prefix.rstrip('/\\')
        # Prepend custom prefix
        path = add_prefix + path
    
    return path

def analyze_flamegraph_strategy_agent(folded_content, source_locations=None, output_base="flamegraph", full_paths=False, excluded_dirs=None, verbose=False):
    """Strategy agent: Branch concentration analysis - checks if the function content is sufficient for analysis"""
    if source_locations is None:
        source_locations = {}
    
    # Parse folded data to get top leaf function percentage directly
    leaf_function_data = {}
    total_samples = 0
    
    for line in folded_content.strip().split('\n'):
        if not line.strip():
            continue
            
        parts = line.rsplit(' ', 1)
        if len(parts) != 2:
            continue
            
        stack_trace = parts[0]
        try:
            sample_count = int(parts[1])
        except ValueError:
            continue
            
        total_samples += sample_count
        
        # Extract leaf function (on-CPU function)
        functions = stack_trace.split(';')
        if functions:
            leaf_function = clean_function_name(functions[-1].strip())
            if leaf_function not in leaf_function_data:
                leaf_function_data[leaf_function] = 0
            leaf_function_data[leaf_function] += sample_count
    
    # Get top percentage from leaf functions
    top_percentage = 0.0
    if leaf_function_data:
        sorted_leaf_functions = sorted(leaf_function_data.items(), key=lambda x: x[1], reverse=True)
        # Filter out kernel/standard functions
        filtered_leaf_functions = [(func, samples) for func, samples in sorted_leaf_functions if not is_filtered_function(func)]
        if filtered_leaf_functions:
            top_samples = filtered_leaf_functions[0][1]
            top_percentage = (top_samples / total_samples) * 100
    
    # Parse folded data to build comprehensive function and stack trace data
    function_data = {}  # leaf function -> sample count
    stack_data = {}     # all functions -> their width (total samples they appear in)
    total_samples = 0
    all_stack_traces = []  # Store all stack traces for ancestor analysis
    
    for line in folded_content.strip().split('\n'):
        if not line.strip():
            continue
            
        parts = line.rsplit(' ', 1)
        if len(parts) != 2:
            continue
            
        stack_trace = parts[0]
        try:
            sample_count = int(parts[1])
        except ValueError:
            continue
            
        total_samples += sample_count
        all_stack_traces.append((stack_trace, sample_count))
        
        # Extract leaf function (on-CPU function)
        functions = stack_trace.split(';')
        if functions:
            leaf_function = clean_function_name(functions[-1].strip())
            if leaf_function not in function_data:
                function_data[leaf_function] = 0
            function_data[leaf_function] += sample_count
        
    
    # Sort functions by sample count and calculate percentages
    sorted_functions = sorted(function_data.items(), key=lambda x: x[1], reverse=True)

    # Filter out Linux kernel functions, mangled names, std:: functions, and unknown functions
    filtered_functions = [(func, samples) for func, samples in sorted_functions 
                         if not is_filtered_function(func) and func != '<unknown>' and 'unknown' not in func.lower()]
    
    if verbose:
        print(f"[*] Filtered {len(sorted_functions) - len(filtered_functions)} functions (kernel/mangled/std::/unknown)")
        print(f"[*] Remaining functions: {len(filtered_functions)}")
    
    # Get top 5 filtered functions
    top_5_functions = filtered_functions[:5]
    result_functions = []
    
    # Get the filtered-out functions that were in top 5 before filtering
    top_5_before_filtering = sorted_functions[:5]
    filtered_out_top_5 = [(func, samples) for func, samples in top_5_before_filtering 
                          if is_filtered_function(func) or func == '<unknown>' or 'unknown' in func.lower()]
    
    # Process top 5 functions
    for func, samples in top_5_functions:
        percentage = (samples / total_samples) * 100
        result_functions.append((func, samples, percentage, "on-CPU"))
        
        # If analysis is needed (top percentage <10%) and function is <10%, find wide ancestor
        if analysis_needed and percentage < 10.0:
            ancestor_result = find_wide_ancestor(func, samples)
            if ancestor_result:
                ancestor_func, ancestor_samples, ancestor_percentage = ancestor_result
                # Add ancestor if not already in results
                if not any(rf[0] == ancestor_func for rf in result_functions):
                    result_functions.append((ancestor_func, ancestor_samples, ancestor_percentage, f"ancestor of {func}"))
    
    # Process filtered-out functions from original top 5 and find their ancestors
    for func, samples in filtered_out_top_5:
        if verbose:
            percentage = (samples / total_samples) * 100
            print(f"[*] Finding ancestor for filtered function: {func} ({percentage:.2f}%)")
        
        ancestor_result = find_wide_ancestor(func, samples)
        if ancestor_result:
            ancestor_func, ancestor_samples, ancestor_percentage = ancestor_result
            # Add ancestor if not already in results
            if not any(rf[0] == ancestor_func for rf in result_functions):
                result_functions.append((ancestor_func, ancestor_samples, ancestor_percentage, f"ancestor of filtered {func}"))
    
    # Create output, filtering out functions with unknown locations
    output_lines = []
    rank = 1
    
    for func, samples, percentage, func_type in result_functions:
        # Skip functions with on-CPU time < 2%
        if percentage < 2.0:
            continue
        
        # Get source location using improved matching
        location = find_best_source_location_match(func, func, source_locations)

        # Skip functions with unknown location
        if not location:
            continue

        # Skip functions in excluded directories
        if excluded_dirs and is_function_in_excluded_dirs(location, excluded_dirs):
            continue

        output_lines.append((func, location))
        
        # Add to JSON output
        rank += 1
    
    
    
    # Save JSON to file
    # json_filename = f"{output_base}_strategy5.json"
    # try:
    #     with open(json_filename, 'w', encoding='utf-8') as f:
    #         json.dump(json_output, f, indent=2)
    #     print(f"[✓] JSON output saved to: {json_filename}")
    # except Exception as e:
    #     print(f"[!] Warning: Could not save JSON to {json_filename}: {e}", file=sys.stderr)
    
    # Create appropriate output message based on whether analysis was needed
    if analysis_needed:
        analysis_method = """ANALYSIS METHOD:
- Took top 5 filtered on-CPU functions
- For each <10% function, recursively searched ancestors until finding one ≥25% wide
- Combined results below"""
        threshold_message = f"Top function percentage: {top_percentage:.2f}% (< 10% threshold - extensive branching detected)"
    else:
        analysis_method = """ANALYSIS METHOD:
- Took top 5 filtered on-CPU functions
- No ancestor analysis needed since top function ≥10%"""
        threshold_message = f"Top function percentage: {top_percentage:.2f}% (≥ 10% threshold - concentrated execution detected)"

    # Return list of tuples: [(function_name, full_path), ...]
    return output_lines



def analyze_flamegraph_strategy_5(folded_content, source_locations=None, output_base="flamegraph", full_paths=False, excluded_dirs=None, verbose=False):
    """Strategy 5: Branch concentration analysis - checks if low top percentage is due to many small functions or few larger ones."""
    if source_locations is None:
        source_locations = {}
    
    # Parse folded data to get top leaf function percentage directly
    leaf_function_data = {}
    total_samples = 0
    
    for line in folded_content.strip().split('\n'):
        if not line.strip():
            continue
            
        parts = line.rsplit(' ', 1)
        if len(parts) != 2:
            continue
            
        stack_trace = parts[0]
        try:
            sample_count = int(parts[1])
        except ValueError:
            continue
            
        total_samples += sample_count
        
        # Extract leaf function (on-CPU function)
        functions = stack_trace.split(';')
        if functions:
            leaf_function = clean_function_name(functions[-1].strip())
            if leaf_function not in leaf_function_data:
                leaf_function_data[leaf_function] = 0
            leaf_function_data[leaf_function] += sample_count
    
    # Get top percentage from leaf functions
    top_percentage = 0.0
    if leaf_function_data:
        sorted_leaf_functions = sorted(leaf_function_data.items(), key=lambda x: x[1], reverse=True)
        # Filter out kernel/standard functions
        filtered_leaf_functions = [(func, samples) for func, samples in sorted_leaf_functions if not is_filtered_function(func)]
        if filtered_leaf_functions:
            top_samples = filtered_leaf_functions[0][1]
            top_percentage = (top_samples / total_samples) * 100
    
    # Check if top percentage is less than 10% - if not, we'll still create JSON but with simpler analysis
    analysis_needed = top_percentage < 10.0
    
    # Parse folded data to build comprehensive function and stack trace data
    function_data = {}  # leaf function -> sample count
    stack_data = {}     # all functions -> their width (total samples they appear in)
    total_samples = 0
    all_stack_traces = []  # Store all stack traces for ancestor analysis
    
    for line in folded_content.strip().split('\n'):
        if not line.strip():
            continue
            
        parts = line.rsplit(' ', 1)
        if len(parts) != 2:
            continue
            
        stack_trace = parts[0]
        try:
            sample_count = int(parts[1])
        except ValueError:
            continue
            
        total_samples += sample_count
        all_stack_traces.append((stack_trace, sample_count))
        
        # Extract leaf function (on-CPU function)
        functions = stack_trace.split(';')
        if functions:
            leaf_function = clean_function_name(functions[-1].strip())
            if leaf_function not in function_data:
                function_data[leaf_function] = 0
            function_data[leaf_function] += sample_count
        
        # Track width of all functions in the stack (how many samples they contribute to)
        for func_raw in functions:
            func = clean_function_name(func_raw.strip())
            if func not in stack_data:
                stack_data[func] = 0
            stack_data[func] += sample_count
    
    # Sort functions by sample count and calculate percentages
    sorted_functions = sorted(function_data.items(), key=lambda x: x[1], reverse=True)

    # Filter out Linux kernel functions, mangled names, std:: functions, and unknown functions
    filtered_functions = [(func, samples) for func, samples in sorted_functions 
                         if not is_filtered_function(func) and func != '<unknown>' and 'unknown' not in func.lower()]
    
    if verbose:
        print(f"[*] Filtered {len(sorted_functions) - len(filtered_functions)} functions (kernel/mangled/std::/unknown)")
        print(f"[*] Remaining functions: {len(filtered_functions)}")
    
    # Get top 5 filtered functions
    top_5_functions = filtered_functions[:5]
    result_functions = []
    
    # Get the filtered-out functions that were in top 5 before filtering
    top_5_before_filtering = sorted_functions[:5]
    filtered_out_top_5 = [(func, samples) for func, samples in top_5_before_filtering 
                          if is_filtered_function(func) or func == '<unknown>' or 'unknown' in func.lower()]
    
    def find_wide_ancestor(target_func, target_samples):
        """Find most immediate ancestor of target_func that has at least 25% width"""
        target_percentage = (target_samples / total_samples) * 100
        
        if verbose:
            print(f"[*] Checking ancestors for {target_func} ({target_percentage:.2f}%)")
        
        # If function is already >= 25%, no need to look for ancestors
        if target_percentage >= 25.0:
            return None
            
        # Find all stack traces containing this function
        containing_traces = []
        for stack_trace, sample_count in all_stack_traces:
            functions = [clean_function_name(f.strip()) for f in stack_trace.split(';')]
            if target_func in functions:
                containing_traces.append((functions, sample_count))
        
        if not containing_traces:
            return None
        
        # For each position this function appears, check its ancestors
        best_ancestor = None
        best_width = 0
        best_depth = float('inf')  # Track call depth to find most immediate ancestor
        
        for functions, sample_count in containing_traces:
            if target_func not in functions:
                continue
                
            func_index = functions.index(target_func)
            
            # Check ancestors (functions earlier in the stack) - start from most immediate
            for i in range(func_index - 1, -1, -1):
                ancestor = functions[i]
                
                # Skip filtered functions for ancestors too
                if is_filtered_function(ancestor):
                    continue
                
                # Calculate ancestor's width
                ancestor_width = stack_data.get(ancestor, 0)
                ancestor_percentage = (ancestor_width / total_samples) * 100
                
                if ancestor_percentage >= 25.0:
                    # Calculate call depth (distance from target function)
                    call_depth = func_index - i
                    
                    # Choose this ancestor if it's more immediate (shorter call depth)
                    # or if it's the same depth but has higher width
                    if (call_depth < best_depth or 
                        (call_depth == best_depth and ancestor_width > best_width)):
                        best_ancestor = ancestor
                        best_width = ancestor_width
                        best_depth = call_depth
                    
                    # Since we're going from most immediate to least immediate,
                    # we can break after finding the first qualifying ancestor in this trace
                    break
        
        if best_ancestor:
            best_percentage = (best_width / total_samples) * 100
            if verbose:
                print(f"[*] Found most immediate ancestor: {best_ancestor} ({best_percentage:.2f}% wide, depth: {best_depth})")
            return (best_ancestor, best_width, best_percentage)
        
        if verbose:
            print(f"[*] No 25%+ ancestor found for {target_func}")
        return None
    
    # Process top 5 functions
    for func, samples in top_5_functions:
        percentage = (samples / total_samples) * 100
        result_functions.append((func, samples, percentage, "on-CPU"))
        
        # If analysis is needed (top percentage <10%) and function is <10%, find wide ancestor
        if analysis_needed and percentage < 10.0:
            ancestor_result = find_wide_ancestor(func, samples)
            if ancestor_result:
                ancestor_func, ancestor_samples, ancestor_percentage = ancestor_result
                # Add ancestor if not already in results
                if not any(rf[0] == ancestor_func for rf in result_functions):
                    result_functions.append((ancestor_func, ancestor_samples, ancestor_percentage, f"ancestor of {func}"))
    
    # Process filtered-out functions from original top 5 and find their ancestors
    for func, samples in filtered_out_top_5:
        if verbose:
            percentage = (samples / total_samples) * 100
            print(f"[*] Finding ancestor for filtered function: {func} ({percentage:.2f}%)")
        
        ancestor_result = find_wide_ancestor(func, samples)
        if ancestor_result:
            ancestor_func, ancestor_samples, ancestor_percentage = ancestor_result
            # Add ancestor if not already in results
            if not any(rf[0] == ancestor_func for rf in result_functions):
                result_functions.append((ancestor_func, ancestor_samples, ancestor_percentage, f"ancestor of filtered {func}"))
    
    # Create output, filtering out functions with unknown locations
    output_lines = []
    rank = 1
    
    for func, samples, percentage, func_type in result_functions:
        # Skip functions with on-CPU time < 2%
        if percentage < 2.0:
            continue
        
        # Get source location using improved matching
        location = find_best_source_location_match(func, func, source_locations)

        # Skip functions with unknown location
        if not location:
            continue

        # Skip functions in excluded directories
        if excluded_dirs and is_function_in_excluded_dirs(location, excluded_dirs):
            continue

        output_lines.append((func, location))
        
        # Add to JSON output
        rank += 1
    
    
    
    # Save JSON to file
    # json_filename = f"{output_base}_strategy5.json"
    # try:
    #     with open(json_filename, 'w', encoding='utf-8') as f:
    #         json.dump(json_output, f, indent=2)
    #     print(f"[✓] JSON output saved to: {json_filename}")
    # except Exception as e:
    #     print(f"[!] Warning: Could not save JSON to {json_filename}: {e}", file=sys.stderr)
    
    # Create appropriate output message based on whether analysis was needed
    if analysis_needed:
        analysis_method = """ANALYSIS METHOD:
- Took top 5 filtered on-CPU functions
- For each <10% function, recursively searched ancestors until finding one ≥25% wide
- Combined results below"""
        threshold_message = f"Top function percentage: {top_percentage:.2f}% (< 10% threshold - extensive branching detected)"
    else:
        analysis_method = """ANALYSIS METHOD:
- Took top 5 filtered on-CPU functions
- No ancestor analysis needed since top function ≥10%"""
        threshold_message = f"Top function percentage: {top_percentage:.2f}% (≥ 10% threshold - concentrated execution detected)"

    # Return list of tuples: [(function_name, full_path), ...]
    return output_lines


def unified_strategy(cmd, output_base, verbose, full_paths, exclude_dir):
    
    try:
        #get the pid of clickhouse
        result = subprocess.run(
            ["pidof", "clickhouse"],
            capture_output=True,
            text=True,
            check=False,
        )

        # Generate flame graph folded file
        pid = int(result.stdout.strip())
        folded_path = run_flamegraph(executable_command=cmd, pid=pid, output_base=output_base)
        
        # Extract source locations from perf.data
        perf_data_path = "perf.data"
        if verbose:
            print("[*] Extracting source locations from perf.data...")
        source_locations = extract_source_locations(perf_data_path, full_paths=full_paths)

        # Read folded file content
        if verbose:
            print("[*] Reading folded file for analysis...")
        folded_content = read_folded_file(folded_path)
        
        # Run the default strategy 5 analysis
        if verbose:
            print("\n" + "="*80)
            print("WIDE PLATEAU IDENTIFICATION")
            print("="*80)
        analysis = analyze_flamegraph_strategy_5(folded_content, source_locations, output_base, full_paths=full_paths, excluded_dirs=exclude_dir, verbose=verbose)
        
        return analysis
        
    except Exception as e:
        print(f"Error in unified_strategy: {e}", file=sys.stderr)
        raise e
