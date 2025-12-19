# Smart Organizer

A command-line file organization tool that uses AI to suggest standardized filenames and directory structures. The tool operates interactively, presenting suggestions for user approval before making changes.

## Requirements

- Python 3.8+
- Dependencies: see `requirements.txt`
- OpenAI API key

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API key:

```
OPENAI_API_KEY=your_api_key_here
```

**Windows (Automated):**

```powershell
powershell -ExecutionPolicy Bypass -File Install-SmartOrganizer.ps1
```

## Usage

**Interactive mode:**

```bash
python smart_organizer.py
```

**Command-line mode:**

```bash
python smart_organizer.py file <path>    # Process a single file
python smart_organizer.py dir <path>     # Process a directory
python smart_organizer.py batch          # Batch processing mode
python smart_organizer.py log            # View operation log
python smart_organizer.py dirs           # Manage directory registry
python smart_organizer.py config         # Configure settings
python smart_organizer.py help           # Display help
```

## Configuration

Settings are stored in `~/.smart_organizer/config.json`:

| Setting | Description | Default |
|---------|-------------|---------|
| `default_base_dir` | Base directory for organized files | `~/Cleanup` |
| `ai_model` | OpenAI model to use | `gpt-4o` |
| `min_confidence_threshold` | Minimum AI confidence for auto-suggestions | `0.7` |
| `default_filename_pattern` | Pattern for renamed files | `{year}-{month}-{category}_{descriptor}_{context}` |
| `default_directory_pattern` | Pattern for new directories | `{category}` |

## Naming Patterns

**Files:** `{year}-{month}-{category}_{descriptor}_{context}.ext`

Example: `2025-03-finance_quarterly-report_q2.pdf`

**Directories:** `{category}`

Example: `FinancialReports`

## Limitations

- Requires active internet connection for AI features
- API usage incurs OpenAI costs
- Large files (>1MB) cannot be previewed
- Binary files show extension only, no content preview

## License

CC0 1.0 - see LICENSE
