---
trigger: always_on
---

# Fail-Fast Coding Guidelines and Invariants

## Core Principle
Write **fail-fast, explicit, non-defensive code**. Code should crash immediately and loudly with clear error messages whenever an assumption, parameter, or invariant is violated. Never write code that silently masks bugs, swallows errors, or falls back to "convenient" defaults.

For example:

### ❌ No Defensive Chained `.get(key, default)`
- **Prohibited**:
  ```python
  # BAD: Masks missing configs, conflates distinct mathematical parameters,
  # and hides typos silently during sweeps.
  target_lambda = config.get("VALUE_LAMBDA", config.get("GAE_LAMBDA", 0.9))
  ```
- **Required**:
  ```python
  # GOOD: Explicit dispatch; crashes immediately if misconfigured
  if critic_type in ["td", "sarsa"]:
      target_lambda = config["VALUE_LAMBDA"]
  elif critic_type in ["e_0", "e_lambda"]:
      target_lambda = config["RETURN_LAMBDA"]
  else:
      raise ValueError(f"Unrecognized critic_type '{critic_type}'")
  ```

### ❌ No Silent Fallback or Exception Swallowing
- Never use bare `except: pass` or `except Exception:` to hide initialization failures or attribute mismatches.
- If an attribute or configuration is missing or malformed, let the exception surface immediately.


### Yes - Explicit Dictionary Access for Required Keys
- Use `config["KEY"]` rather than `config.get("KEY")` for any parameter that the algorithm mathematically depends on (e.g., `GAMMA`, `LR`, `CRITIC_LR`, `NUM_STEPS`, `NUM_ENVS`). If a new hyperparamter is being added, put a reasonable default in the base config file
- Only use `.get(key, default)` for truly optional, aesthetic toggles (e.g., verbose logging flags) where the default is strictly defined and invariant to the algorithm chosen.
