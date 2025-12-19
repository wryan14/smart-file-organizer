# Smart Organizer

Uses AI to generate standardized filenames and organize files into categorized directories based on user-provided descriptions. Operates interactively, presenting suggestions for approval before making changes. Maintains a registry of known directories to suggest appropriate placement for new files.

## Requirements

- Python 3.8+
- Dependencies: see `requirements.txt`
- OpenRouter API key (or OpenAI API key for direct access)

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

Get an API key at [openrouter.ai](https://openrouter.ai/). OpenRouter provides access to models from OpenAI, Anthropic, Google, Meta, and others through a single API.

Windows users can run the automated installer, which creates desktop shortcuts and handles dependency installation:

```powershell
powershell -ExecutionPolicy Bypass -File Install-SmartOrganizer.ps1
```

## Usage

Run without arguments for interactive mode:

```bash
python smart_organizer.py
```

Or use direct commands:

```bash
python smart_organizer.py file <path>    # Process a single file
python smart_organizer.py dir <path>     # Process a directory
python smart_organizer.py batch          # Batch processing mode
python smart_organizer.py log            # View operation log
python smart_organizer.py dirs           # Manage directory registry
python smart_organizer.py config         # Configure settings
```

## Configuration

Settings are stored in `~/.smart_organizer/config.json`:

| Setting | Description | Default |
|---------|-------------|---------|
| `default_base_dir` | Base directory for organized files | `~/Cleanup` |
| `api_provider` | `openrouter` or `openai` | `openrouter` |
| `api_model` | Model identifier (OpenRouter format: `provider/model`) | `openai/gpt-4o` |
| `min_confidence_threshold` | Minimum confidence for auto-suggesting existing directories | `0.7` |
| `default_filename_pattern` | Pattern for renamed files | `{year}-{month}-{category}_{descriptor}_{context}` |
| `default_directory_pattern` | Pattern for new directories | `{category}` |

### Model Examples

```json
{
  "api_model": "openai/gpt-4o"
}
```

```json
{
  "api_model": "anthropic/claude-sonnet-4"
}
```

```json
{
  "api_model": "google/gemini-2.0-flash-exp"
}
```

See [openrouter.ai/models](https://openrouter.ai/models) for available models.

### Using OpenAI Directly

To bypass OpenRouter and use OpenAI directly:

```json
{
  "api_provider": "openai",
  "api_model": "gpt-4o"
}
```

Set `OPENAI_API_KEY` in your `.env` file instead of `OPENROUTER_API_KEY`.

## Limitations

- Requires active internet connection for AI features
- API usage incurs costs (varies by model and provider)
- Files larger than 1MB cannot be previewed (binary files show extension only)
- Directory suggestions require sufficient registered directories for meaningful matching

## License

CC0 1.0 - see LICENSE
