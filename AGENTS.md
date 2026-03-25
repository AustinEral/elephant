# Elephant

Always use the `elephant` MCP server when it is available and the task would benefit from persistent memory.

Use Elephant for:

- storing user preferences, project conventions, durable facts, and important task outcomes with `retain`
- recalling prior project or user context before non-trivial tasks with `recall`
- answering or synthesizing memory-grounded questions with `reflect`

Do not use Elephant for:

- ephemeral scratch work that should not be remembered
- questions that are fully answerable from the current repository state without prior memory

If the `elephant` MCP server is unavailable, continue normally and note that memory integration is unavailable.
