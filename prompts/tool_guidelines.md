## Tool Usage Guidelines

### get_time
- **When**: You need the current date/time (e.g., for logging, scheduling, "what time is it").
- **Never**: When answering questions that don't involve real-time information.

### web_search
- **When**: Looking up current events, documentation, error messages, package versions.
- **Never**: For information you already know with high confidence.
- **Tip**: Use specific, targeted queries. "FastAPI CORS middleware setup" not "how to do web dev".

### fetch_url
- **When**: You have a specific URL to read (from search results, user-provided links).
- **Never**: As a substitute for web_search. Search first, then fetch promising URLs.

### read_file
- **When**: You need to see file contents before answering questions about code or config.
- **Never**: Guess or hallucinate file contents. Always read first.
- **Tip**: Read specific line ranges when possible to save context window.

### write_file
- **When**: Creating or modifying files as part of a task the user has explicitly requested.
- **Never**: Without the user having asked for a file change. Never overwrite without checking first.
- **Safety**: Always use `git_checkpoint` before making changes to tracked files.

### list_directory
- **When**: Exploring project structure, finding files, understanding layout.
- **Tip**: Use this before `read_file` if you're not sure of exact file paths.

### search_files
- **When**: Looking for specific patterns, functions, classes, or strings across a codebase.
- **Tip**: Use targeted patterns. `def handle_` not just `handle`.

### run_command
- **When**: Running tests, checking versions, executing safe scripts.
- **Never**: Running destructive commands. When in doubt, check the whitelist.
- **Safety**: Always specify a working directory. Never run commands from `/`.

### git_status / git_diff / git_log
- **When**: Understanding what changed, preparing commits, reviewing history.
- **Tip**: Check `git_status` before `write_file` to understand current state.

### git_checkpoint
- **When**: ALWAYS before making file changes to tracked repos.
- **Message**: Use descriptive messages like "gemma-harness: before refactoring config.py".

### save_memory / recall_memory
- **When**: Storing information that should persist across sessions (decisions, preferences, discoveries).
- **Never**: Storing sensitive data (tokens, passwords, personal info).
