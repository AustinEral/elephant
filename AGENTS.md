# Elephant

Use the `elephant` MCP server when it is available and the task would benefit from persistent memory.

If `.elephant/bank_id` exists in the repository, treat its contents as the default Elephant bank ID for this repo and use that bank unless the user explicitly asks for a different one.

Use Elephant for storing, recalling, and reflecting on durable user, project, and workflow memory.
Do not use Elephant for ephemeral scratch work or questions fully answerable from the current repository state.

If the `elephant` MCP server is unavailable, continue normally and note that memory integration is unavailable.
