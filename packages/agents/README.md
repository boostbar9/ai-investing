# packages/agents

Four agents wired into a LangGraph + Temporal workflow.

| Agent | Job | Primary | Backup | Input | Output |
|---|---|---|---|---|---|
| Research | News, macro | DeepSeek R1 | Qwen 2.5 | `ResearchInput` | `ResearchOutput` |
| Strategy | Signals | Qwen 2.5 | Llama 3.3 | `StrategyInput` | `StrategyOutput` |
| Risk | Exposure, halt | DeepSeek R1 | Mistral Large | `RiskInput` | `RiskOutput` |
| Execution | Order routing | Llama 3.3 | Mistral Large | `ExecutionInput` | `ExecutionOutput` |

Schemas live in `packages/shared/schemas.py`.
