# Rotation log

Append-only. One entry per secret per rotation. Newest at top.

| Date (UTC) | Secret               | Config(s)     | Operator | Reason    | Audit ID |
|------------|----------------------|---------------|----------|-----------|----------|
| —          | _(no rotations yet)_ | —             | —        | —         | —        |

## How to add an entry

```bash
# After completing a rotation, capture the audit_id from /audit/security/...
# and add a row above this section. Commit with a message like:
git commit -m "chore(security): log rotation of ALPACA_PAPER_SECRET"
```
