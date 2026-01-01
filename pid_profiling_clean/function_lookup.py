import argparse, re
from pathlib import Path
from typing import Optional, Tuple
import subprocess


def find_paren_close(src:str, code:list[bool], i_open:int)->int:
    assert src[i_open]=='('
    depth=0; j=i_open; n=len(src)
    while j<n:
        if code[j]:
            ch=src[j]
            if ch=='(':
                depth+=1
            elif ch==')':
                depth-=1
                if depth==0: return j
        j+=1
    return -1
def find_matching(src:str, code:list[bool], i_open:int, open_ch='{', close_ch='}')->int:
    assert src[i_open]==open_ch
    depth=0; j=i_open; n=len(src)
    while j<n:
        if code[j]:
            ch=src[j]
            if ch==open_ch: depth+=1
            elif ch==close_ch:
                depth-=1
                if depth==0: return j
        j+=1
    return -1
def next_code_pos(src:str, code:list[bool], idx:int, forward=True)->int:
    n=len(src); j=idx
    if forward:
        while j<n and (not code[j] or src[j].isspace()): j+=1
    else:
        while j>=0 and (not code[j] or src[j].isspace()): j-=1
    return j
def build_code_mask(src: str) -> list[bool]:
    n = len(src)
    code = [True] * n
    i = 0
    def mark(a,b):
        a=max(0,a); b=min(n,b)
        for k in range(a,b): code[k]=False

    while i < n:
        c = src[i]
        # // line comment
        if c=='/' and i+1<n and src[i+1]=='/':
            j=i+2
            while j<n and src[j] != '\n': j+=1
            mark(i,j); i=j; continue
        # /* block comment */
        if c=='/' and i+1<n and src[i+1]=='*':
            j=i+2
            while j+1<n and not (src[j]=='*' and src[j+1]=='/'): j+=1
            j=min(j+2,n)
            mark(i,j); i=j; continue
        # raw string? ([u8|u|U|L]?) R"delim( ... )delim"
        def raw_at(k:int)->Optional[Tuple[int,str]]:
            p=k
            if p<n and src[p] in 'uUL':
                if src[p:p+2]=='u8': p+=2
                else: p+=1
            if p<n and src[p]=='R' and p+1<n and src[p+1]=='"':
                d0=p+2; q=d0
                while q<n and src[q] not in '(\n': q+=1
                if q<n and src[q]=='(':
                    return (k, src[d0:q])
            return None
        rp = raw_at(i)
        if rp:
            _, delim = rp
            close = f"){delim}\""
            idx = src.find(close, i)
            j = n if idx==-1 else idx+len(close)
            mark(i,j); i=j; continue
        # "string"
        if c=='"':
            j=i+1; esc=False
            while j<n:
                ch=src[j]
                if esc: esc=False; j+=1; continue
                if ch=='\\': esc=True; j+=1; continue
                if ch=='"': j+=1; break
                j+=1
            mark(i,j); i=j; continue
        # 'char'
        if c=="'":
            j=i+1; esc=False
            while j<n:
                ch=src[j]
                if esc: esc=False; j+=1; continue
                if ch=='\\': esc=True; j+=1; continue
                if ch=="'": j+=1; break
                j+=1
            mark(i,j); i=j; continue
        i+=1
    return code
def find_matching_bracket(s: str, start_index: int, open_char: str = '<', close_char: str = '>') -> int:
    """
    Finds the matching closing bracket for the open bracket at start_index.
    Returns -1 if no matching bracket is found.
    """
    level = 1
    for i in range(start_index + 1, len(s)):
        if s[i] == open_char:
            level += 1
        elif s[i] == close_char:
            level -= 1
            if level == 0:
                return i
    return -1
def find_function_definition(src:str, name:str, qualifier:Optional[str]=None, occurrence:int=1):
    code = build_code_mask(src)
    if qualifier:
        pat = re.compile(rf"{re.escape(qualifier)}{re.escape(name)}\s*\(", re.M)
    else:
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(", re.M)
    count=0
    for m in pat.finditer(src):
        i_name = m.start()
        if not code[i_name]: continue
        i_paren = src.find('(', m.start())
        if i_paren==-1 or not code[i_paren]: continue
        j_paren = find_paren_close(src, code, i_paren)
        if j_paren==-1: continue
        k = next_code_pos(src, code, j_paren+1, True)
        if k>=len(src): continue
        # handle ctor initializer list
        if src[k]==':':
            p=k+1; found=None
            while p<len(src):
                if code[p]:
                    if src[p]=='{': found=('{{',p); break
                    if src[p]==';': found=(';',p); break
                p+=1
            if not found or found[0]==';': continue
            k = found[1]
        if src[k] not in '{;':
            p=k; found=None
            while p<len(src):
                if code[p]:
                    if src[p]=='{': found=('{{',p); break
                    if src[p]==';': found=(';',p); break
                p+=1
            if not found or found[0]==';': continue
            k = found[1]
        if src[k] != '{': continue
        brace_open = k
        brace_close = find_matching(src, code, brace_open, '{','}')
        if brace_close==-1: continue
        
        # --- START: NEW LOGIC TO SKIP RETURN-ONLY FUNCTIONS ---
        body_content_chars = []
        for i in range(brace_open + 1, brace_close):
            if code[i]:
                body_content_chars.append(src[i])
        
        # 2. Create a clean, stripped string of the body.
        clean_body = "".join(body_content_chars).strip()
        
        # 3. Check if it's a single return statement.
        is_return_only = (
            clean_body.startswith("return") and  # Starts with "return"
            clean_body.endswith(";") and          # Ends with ";"
            clean_body.count(";") == 1            # Has exactly one ";"
        )
        
        if is_return_only:
            continue  # Skip this function, it's just a wrapper
            
        # --- END: NEW LOGIC ---
        # choose start as the start of the signature line (simple)
        start = src.rfind('\n', 0, i_name) + 1
        count+=1
        if count==occurrence:
            return start, brace_open, brace_close
    return None

def fallback_function_definition_finder(name:str, executable:str, qualifier:Optional[str]=None):
    BIN = executable
    SYM = f"{qualifier}{name}" if qualifier else name
    
    # Option 1: Using shell=True (simpler for pipes)
    demangling_command = f"nm -anC '{BIN}' | rg -F '{SYM}'"
    result = subprocess.run(
        demangling_command,
        shell=True,  # Required for pipe to work
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError("Function not found by demangling tool")
    #hex address
    address = result.stdout.split(" ")[0]
    #addr2line command
    addr2line_command = f"addr2line -Cfipe '{BIN}' 0x'{address}'"
    result = subprocess.run(
        addr2line_command,
        shell=True,  # Required for pipe to work
        capture_output=True,
        text=True,
        check=False
    )
    func_location = (result.stdout.split(" at ")[1]).split(":")[0]
    return func_location

def parse_short_function_name(full_name: str) -> str:
    """
    Parses the short function name from its full, qualified signature.
    
    Handles:
    - Namespaces: DB::compress
    - Templated functions: DB::executeImplBatch<...>
    - Members of templated classes: DB::MyClass<T>::myFunc
    - Anonymous namespaces: DB::(anonymous namespace)::compress
    - Operators: DB::MyClass::operator<
    """
    # Clean up any surrounding quotes
    full_name = full_name.strip(" '")
    
    # 1. Split the full name by '::', but respect templates
    parts = split_top_level(full_name, '::')
    
    # 2. The name we want is the last part
    if not parts:
        return ""
        
    last_part = parts[-1].strip()
    
    # 3. Now, we just need to check if this *last part* has its own
    #    template. We can't just use split('<')[0] because of 'operator<'
    
    first_template_char = last_part.find('<')
    
    if first_template_char == -1:
        # No '<' found, so it's a simple name
        # e.g., "compress" or "vectorConstant"
        return last_part

    # 4. A '<' was found. Check if it's a balanced template
    #    that goes to the end of the string.
    
    # Find the matching closing bracket
    last_template_char = find_matching_bracket(last_part, first_template_char, '<', '>')
    
    # We strip() the end to allow for trailing whitespace
    if last_template_char == len(last_part.rstrip()) - 1:
        # The '<...>' is balanced and goes to the end.
        # This is a templated function, e.g., "executeImplBatch<...>"
        # The name is everything before the '<'
        return last_part[:first_template_char]
    else:
        # The '<' is part of the name (e.g., "operator<") or
        # is otherwise not a function template (malformed?).
        return last_part
def split_top_level(s: str, delimiter: str = '::') -> list[str]:
    """
    Splits a string by a delimiter, but only when the delimiter is
    at the "top level" (i.e., not inside balanced <> brackets).
    """
    parts = []
    level = 0
    current_part_start = 0
    i = 0
    
    while i < len(s):
        if s[i] == '<':
            level += 1
        elif s[i] == '>':
            if level > 0:
                level -= 1
        elif s.startswith(delimiter, i) and level == 0:
            parts.append(s[current_part_start:i])
            i += len(delimiter)
            current_part_start = i
            continue
        i += 1
    
    parts.append(s[current_part_start:])
    return parts


def function_lookup(function_name, file_path, executable):
    short_function_name = parse_short_function_name(function_name)
    print(short_function_name)
    with open(file_path, 'r') as f:
        source_code = f.read()
    f.close()
    result = find_function_definition(source_code, short_function_name)
    
    if result:
        start, brace_open, brace_close = result
        print(f"Function '{function_name}' () found:")
        return file_path
    else:
        print(f"Function '{function_name}' () not found using traditional way. Trying alternative")
        alternative_file_path = fallback_function_definition_finder(function_name, executable)
        print(alternative_file_path)
        with open(alternative_file_path, 'r') as f:
            source_code = f.read()      
        result = find_function_definition(source_code, short_function_name) 
        if result:
            start, brace_open, brace_close = result
            print(f"Function '{function_name}' () found:")
            return alternative_file_path
        else:
            print(result)
            print("standart and alternative methods of function lookup failed, skipping this function")
            return None

