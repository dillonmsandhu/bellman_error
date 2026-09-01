import os
import re
import glob

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern 1: R_π_s = jnp.einsum("sa,sa->s", Pi, evaluator.R)
    # R_π = P_π @ R_π_s
    # (with some optional comments in between)
    p1 = re.compile(r'R_π_s\s*=\s*jnp\.einsum\("sa,sa->s",\s*([^,]+),\s*evaluator\.R\)\s*(?:#[^\n]*\n\s*)*R_π\s*=\s*P_π\s*@\s*R_π_s')
    
    # Pattern 2: R_pi_s = jnp.einsum("sa,sa->s", old_pi_full, evaluator.R)
    # R_pi = P_pi @ R_pi_s
    p2 = re.compile(r'R_pi_s\s*=\s*jnp\.einsum\("sa,sa->s",\s*([^,]+),\s*evaluator\.R\)\s*(?:#[^\n]*\n\s*)*R_pi\s*=\s*P_pi\s*@\s*R_pi_s')
    
    # Pattern 3: R_pi_delayed = jnp.einsum("sa,sa->s", old_pi_full, evaluator.R)
    # R_pi_shifted = P_pi @ R_pi_delayed
    p3 = re.compile(r'R_pi_delayed\s*=\s*jnp\.einsum\("sa,sa->s",\s*([^,]+),\s*evaluator\.R\)\s*(?:#[^\n]*\n\s*)*R_pi_shifted\s*=\s*P_pi\s*@\s*R_pi_delayed')

    orig_content = content
    content = p1.sub(lambda m: f'R_π = jnp.einsum("sa,sam,sam->s", {m.group(1)}, P, evaluator.R)', content)
    content = p2.sub(lambda m: f'R_pi = jnp.einsum("sa,sam,sam->s", {m.group(1)}, P, evaluator.R)', content)
    content = p3.sub(lambda m: f'R_pi_shifted = jnp.einsum("sa,sam,sam->s", {m.group(1)}, P, evaluator.R)', content)

    if content != orig_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")

for root, _, files in os.walk('.'):
    if 'archive' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            process_file(os.path.join(root, file))
