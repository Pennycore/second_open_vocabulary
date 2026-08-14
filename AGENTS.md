# Sol planner / Terra executor policy

For tasks that materially combine planning with implementation, experiment execution, migration, or other state-changing work, use this model-routing policy unless the user explicitly requests a different model or workflow:

- The main task runs on `gpt-5.6-sol` and owns problem analysis, research or implementation strategy, risk review, protocol or plan freeze, task decomposition, and final verification.
- Do not begin state-changing execution until the plan and its safety boundaries are sufficiently explicit for the task.
- After the plan is frozen, delegate concrete execution to a bounded subagent using `gpt-5.6-terra`. Choose `medium` reasoning for routine mechanical work and `high` for debugging, migration, or experiment execution.
- Give Terra the frozen plan, exact scope, allowed paths, stop conditions, required evidence, and prohibited changes. Terra must not silently alter the design, tune frozen experimental choices, broaden scope, overwrite protected artifacts, or silently retry failed protected outputs.
- Terra reports commands or actions taken, outputs, tests, hashes or other verification evidence, and anomalies back to Sol. Sol performs the final review and communicates the outcome to the user.
- If execution reveals that the frozen plan must change, stop the executor and return the decision to Sol before continuing.
- Do not spawn a Terra executor for a simple factual answer, explanation, status report, or small read-only inspection where delegation adds no value.
- If model-specific subagent routing is unavailable, state that limitation; do not claim Terra executed the work.

An explicit user instruction always overrides this default routing policy.
