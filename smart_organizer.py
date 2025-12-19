#!/usr/bin/env python3
"""AI-powered file and directory organization utility.

Provides interactive CLI for renaming files and directories using
AI-generated suggestions, organizing files into categorized directories,
and maintaining an operation log for all changes.
"""

import sys
import os
from pathlib import Path
import json
import shutil
import datetime
import logging
from typing import Dict, List, NamedTuple, Optional, Union
import re

# Rich for beautiful CLI
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.markdown import Markdown

# For OpenAI API
from openai import OpenAI
from dotenv import load_dotenv

# Initialize console
console = Console()

# Set up directories and constants
CONFIG_DIR = Path.home() / ".smart_organizer"
CONFIG_DIR.mkdir(exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.json"
DIRECTORY_REGISTRY_PATH = CONFIG_DIR / "directories.json"
LOG_FILE = CONFIG_DIR / "operation_log.jsonl"

# Configure logging
logging.basicConfig(
    filename=CONFIG_DIR / "smart_organizer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Prompt templates (single source of truth for naming conventions)
FILENAME_PROMPT_TEMPLATE = """Given a file and description, generate a structured filename.
Format: {pattern}

Where:
- {{year}}-{{month}} should be the current date ({year_month}) unless a specific date is mentioned
- {{category}} is a single word representing the domain (e.g., finance, research, project)
- {{descriptor}} is 2-3 words joined by hyphens that capture the main topic
- {{context}} is 1-2 key identifying words

Guidelines:
- The filename should be meaningful and describe the content
- Use kebab-case (hyphens) within descriptor and context
- Use underscores between pattern elements
- Preserve significant identifiers from original filename
- Use lowercase except for proper nouns
- Do not include the file extension

Original: {original}
Description: {description}

Return ONLY the filename in the exact format: {{year}}-{{month}}-{{category}}_{{descriptor}}_{{context}}"""

DIRNAME_PROMPT_TEMPLATE = """Given a directory and description, generate a simple, categorical directory name.
Format: {pattern}

Guidelines:
- The directory name should be a simple, clear category (e.g., Documents, ProjectFiles, FinancialReports)
- Use PascalCase (capitalize each word, no spaces)
- Keep the name concise but descriptive (typically 1-2 words)
- The name should be broad enough to group similar files, not overly specific

Original: {original}
Description: {description}

Return ONLY the directory name as a single PascalCase word or PascalCase compound word."""


class DirectorySuggestion(NamedTuple):
    """Result of directory suggestion for file organization."""
    path: Path
    is_new: bool


# Default configuration
DEFAULT_CONFIG = {
    "max_preview_lines": 10,
    "max_directories_to_show": 10,
    "min_confidence_threshold": 0.7,
    "default_base_dir": str(Path.home() / "Cleanup"),
    "api_provider": "openrouter",
    "api_model": "openai/gpt-4o",
    "skip_confirm_for_move": False,
    "default_directory_pattern": "{category}",
    "default_filename_pattern": "{year}-{month}-{category}_{descriptor}_{context}",
}


class SmartOrganizer:
    """Interactive file organization tool with AI-powered naming suggestions.

    Manages file and directory renaming, organization into categorized
    directories, and maintains a registry of known directories for
    intelligent placement suggestions.
    """

    def __init__(self):
        """Initialize configuration, directory registry, and AI client."""
        self.config = self._load_config()
        self._ensure_directory_registry()
        self._setup_ai_client()
        self.operation_log = []
        self.batch_queue = []
        
    def _load_config(self) -> Dict:
        """Load configuration or create default if not exists."""
        if not CONFIG_PATH.exists():
            with open(CONFIG_PATH, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            return DEFAULT_CONFIG
        
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)

                # Migrate renamed keys
                updated = False
                if "ai_model" in config and "api_model" not in config:
                    config["api_model"] = config.pop("ai_model")
                    updated = True

                # Add any new default keys
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                        updated = True

                if updated:
                    with open(CONFIG_PATH, "w") as f:
                        json.dump(config, f, indent=2)

                return config
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON in config file: {e}, using defaults")
            return DEFAULT_CONFIG
        except PermissionError as e:
            logging.warning(f"Cannot read config file: {e}, using defaults")
            return DEFAULT_CONFIG

    def _ensure_directory_registry(self):
        """Ensure the directory registry exists and is valid."""
        if not DIRECTORY_REGISTRY_PATH.exists():
            with open(DIRECTORY_REGISTRY_PATH, "w") as f:
                json.dump({"directories": []}, f, indent=2)
    
    def _load_directory_registry(self) -> List[Dict]:
        """Load the directory registry."""
        try:
            with open(DIRECTORY_REGISTRY_PATH, "r") as f:
                data = json.load(f)
                return data.get("directories", [])
        except Exception as e:
            console.print(f"[red]Error loading directory registry: {e}[/]")
            return []
    
    def _save_directory_registry(self, directories: List[Dict]):
        """Save the directory registry."""
        try:
            with open(DIRECTORY_REGISTRY_PATH, "w") as f:
                json.dump({"directories": directories}, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving directory registry: {e}[/]")
    
    def _register_directory(self, path: Path, category: str, description: str):
        """Register a directory in the registry."""
        directories = self._load_directory_registry()
        
        # Check if directory already exists
        for dir_entry in directories:
            if dir_entry["path"] == str(path):
                # Update existing entry
                dir_entry["category"] = category
                dir_entry["description"] = description
                dir_entry["last_updated"] = datetime.datetime.now().isoformat()
                self._save_directory_registry(directories)
                return
        
        # Add new directory
        directories.append({
            "path": str(path),
            "name": path.name,
            "category": category,
            "description": description,
            "created": datetime.datetime.now().isoformat(),
            "last_updated": datetime.datetime.now().isoformat(),
        })
        
        self._save_directory_registry(directories)
    
    def _setup_ai_client(self):
        """Set up the AI client based on configured provider.

        Supports OpenRouter (default) and direct OpenAI. OpenRouter provides
        access to multiple model providers through a single API.

        Raises:
            SystemExit: When API key is missing or client initialization fails
        """
        load_dotenv()
        provider = self.config.get("api_provider", "openrouter")

        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            env_var_name = "OPENROUTER_API_KEY"
            base_url = "https://openrouter.ai/api/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            env_var_name = "OPENAI_API_KEY"
            base_url = None

        if not api_key:
            logging.error(f"{env_var_name} not found in environment or .env file")
            console.print(Panel(
                f"[red]{env_var_name} not found in environment or .env file.[/]\n"
                f"Set your API key to use AI features.",
                title="API Key Error"
            ))
            sys.exit(1)

        try:
            if base_url:
                self.client = OpenAI(base_url=base_url, api_key=api_key)
            else:
                self.client = OpenAI(api_key=api_key)
            self.model = self.config["api_model"]
        except (ValueError, TypeError) as e:
            logging.error(f"AI client initialization failed: {e}")
            console.print(Panel(
                f"[red]AI client initialization failed: {e}[/]\n"
                f"Verify {env_var_name} is valid.",
                title="AI Setup Error"
            ))
            sys.exit(1)

    def _get_ai_completion(self, prompt: str, messages=None, **kwargs) -> Optional[str]:
        """Get completion from the AI model.

        Returns:
            AI response string, or None if request fails
        """
        try:
            if messages is None:
                messages = [{"role": "system", "content": prompt}]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"AI completion request failed: {e}")
            return None
            
    def _batch_add_request(self, file_path: Path, description: str, operation_type: str):
        """Add a request to the batch processing queue."""
        self.batch_queue.append({
            "file_path": file_path,
            "description": description,
            "operation_type": operation_type,
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    def _extract_file_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from a file."""
        if not file_path.exists():
            return {"error": "File not found"}
        
        try:
            return {
                "name": file_path.name,
                "extension": file_path.suffix,
                "size_bytes": file_path.stat().st_size,
                "size_human": self._format_size(file_path.stat().st_size),
                "modified": datetime.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "created": datetime.datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            }
        except Exception as e:
            logging.error(f"Error extracting metadata for {file_path}: {e}")
            return {
                "name": file_path.name,
                "error": f"Error accessing file: {e}"
            }

    def _open_file(self, file_path: Path):
        """Open the file in the default application."""
        try:
            # Cross-platform way to open files with default application
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                subprocess.run(['open', str(file_path)], check=True)
            elif system == 'Windows':
                os.startfile(str(file_path))
            else:  # Linux and other Unix-like
                subprocess.run(['xdg-open', str(file_path)], check=True)
                
            console.print(f"[green]Opened {file_path.name} in default application[/]")
            return True
        except Exception as e:
            console.print(f"[red]Error opening file: {e}[/]")
            return False

    def _open_directory(self, dir_path: Path):
        """Open a directory in the file explorer."""
        try:
            # Cross-platform way to open directories with default file explorer
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                subprocess.run(['open', str(dir_path)], check=True)
            elif system == 'Windows':
                os.startfile(str(dir_path))
            else:  # Linux and other Unix-like
                subprocess.run(['xdg-open', str(dir_path)], check=True)
                
            console.print(f"[green]Opened {dir_path.name} in file explorer[/]")
            return True
        except Exception as e:
            console.print(f"[red]Error opening directory: {e}[/]")
            return False

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"

    def _generate_file_preview(self, file_path: Path) -> str:
        """Generate a preview of the file content."""
        try:
            if not file_path.exists():
                return "[File not found]"
            
            # Don't try to preview large files
            if file_path.stat().st_size > 1024 * 1024:  # 1MB
                return "[File too large for preview]"
            
            file_type = file_path.suffix.lower()
            
            # Handle text files
            if file_type in [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".csv", ".xml", ".yaml", ".yml"]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()[:self.config["max_preview_lines"]]
                return "".join(lines)
            
            # Handle binary files
            return f"[Binary file with extension {file_type}]"
        except Exception as e:
            return f"[Error generating preview: {e}]"

    def _log_operation(self, operation_type: str, details: Dict):
        """Log an operation to the operation log."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": operation_type,
            **details
        }
        
        self.operation_log.append(entry)
        
        # Also write to the log file
        try:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not write to log file: {e}[/]")
            logging.error(f"Failed to write to log file: {e}")

    def _get_standardized_filename(self, original_name: str, description: str) -> Optional[str]:
        """Use AI to generate a standardized filename.

        Returns:
            Standardized filename, "DELETE" if deletion requested, or None on error
        """
        if description.strip().lower() == "del":
            return "DELETE"

        year_month = datetime.datetime.now().strftime("%Y-%m")
        prompt = FILENAME_PROMPT_TEMPLATE.format(
            pattern=self.config["default_filename_pattern"],
            year_month=year_month,
            original=original_name,
            description=description
        )

        result = self._get_ai_completion(prompt)
        if result and result != "DELETE":
            original_ext = Path(original_name).suffix
            if original_ext and not result.endswith(original_ext):
                result += original_ext

        return result

    def _get_standardized_dirname(self, original_path: Path, description: str) -> Optional[str]:
        """Use AI to generate a standardized directory name.

        Returns:
            Standardized directory name, "DELETE" if deletion requested, or None on error
        """
        if description.strip().lower() == "del":
            return "DELETE"

        prompt = DIRNAME_PROMPT_TEMPLATE.format(
            pattern=self.config["default_directory_pattern"],
            original=original_path.name,
            description=description
        )

        return self._get_ai_completion(prompt)

    def _suggest_target_directory(self, file_path: Path, new_name: str, description: str) -> DirectorySuggestion:
        """Suggest a target directory for the file based on existing directories."""
        # Get all registered directories
        directories = self._load_directory_registry()
        
        if not directories:
            # If no directories exist yet, suggest creating a new one
            return self._suggest_new_directory(file_path, new_name, description)
        
        # Create a prompt to find the best match
        prompt = f"""I need to organize a file into the most appropriate existing directory.

File info:
- Original name: {file_path.name}
- New standardized name: {new_name}
- Description: {description}

Available directories:
"""

        for i, directory in enumerate(directories, 1):
            prompt += f"{i}. Name: {directory['name']}\n   Path: {directory['path']}\n   Category: {directory['category']}\n   Description: {directory['description']}\n\n"

        prompt += """Based on the file information and available directories, please:
1. Determine the most appropriate existing directory for this file
2. Provide your reasoning
3. If none of the existing directories are suitable, recommend creating a new directory

Format your response as:
DIRECTORY_NUMBER: [number of the best directory, or "NEW" if a new directory is needed]
CONFIDENCE: [a decimal between 0 and 1 indicating your confidence]
REASONING: [brief explanation of your choice]
NEW_DIRECTORY_SUGGESTION: [if recommending a new directory, suggest a name in the format year-month-category_purpose]"""

        try:
            result = self._get_ai_completion(prompt)
            
            # Parse the AI response
            directory_number = None
            confidence = 0
            reasoning = ""
            new_directory = None
            
            for line in result.split("\n"):
                line = line.strip()
                if line.startswith("DIRECTORY_NUMBER:"):
                    value = line.split(":", 1)[1].strip()
                    if value.isdigit():
                        directory_number = int(value)
                    elif value.upper() == "NEW":
                        directory_number = "NEW"
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        confidence = 0
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                elif line.startswith("NEW_DIRECTORY_SUGGESTION:"):
                    new_directory = line.split(":", 1)[1].strip()
            
            # Determine if we should use an existing directory or create a new one
            if (directory_number and 
                directory_number != "NEW" and 
                1 <= directory_number <= len(directories) and
                confidence >= self.config["min_confidence_threshold"]):
                
                # Use existing directory
                chosen_dir = directories[directory_number - 1]
                target_dir = Path(chosen_dir["path"])
                
                # Display the choice
                console.print(Panel(
                    f"[green]Selected existing directory:[/] {chosen_dir['name']}\n"
                    f"[blue]Path:[/] {chosen_dir['path']}\n"
                    f"[blue]Confidence:[/] {confidence:.2f}\n"
                    f"[blue]Reasoning:[/] {reasoning}",
                    title="Directory Suggestion"
                ))
                
                return DirectorySuggestion(target_dir, False)
            else:
                # Suggest creating a new directory
                return self._suggest_new_directory(file_path, new_name, description, suggested_name=new_directory)
                
        except Exception as e:
            console.print(f"[red]Error suggesting target directory: {e}[/]")
            return DirectorySuggestion(file_path.parent, False)

    def _suggest_new_directory(self, file_path: Path, new_name: str, description: str, suggested_name=None) -> DirectorySuggestion:
        """Suggest creating a new directory."""
        base_dir = Path(self.config["default_base_dir"])
        
        if suggested_name is None:
            # Generate a new directory name
            year_month = datetime.datetime.now().strftime("%Y-%m")
            
            prompt = f"""Given a file, suggest an appropriate new directory name.
Format: {{Category}}

Guidelines:
- The directory name should be a simple, clear category (e.g., "Documents", "ProjectFiles", "FinancialReports")
- Use PascalCase (capitalize each word, no spaces)
- Keep the name concise but descriptive (typically 1-2 words)
- The name should be broad enough to group similar files, not overly specific

File info:
- Original name: {file_path.name}
- New standardized name: {new_name}
- Description: {description}

Return ONLY the directory name as a single PascalCase word or PascalCase compound word."""

            try:
                suggested_name = self._get_ai_completion(prompt)
            except Exception as e:
                console.print(f"[red]Error generating directory name: {e}[/]")
                suggested_name = f"{year_month}-files_misc"
        
        # Create a friendly reason
        prompt = f"""Explain briefly why this file should go in a new directory rather than existing ones.
The directory suggestion is: {suggested_name}
The file is: {new_name}
The file description is: {description}

Keep it short (1-2 sentences). Focus on organization benefits."""

        try:
            reasoning = self._get_ai_completion(prompt)
        except Exception:
            reasoning = "This appears to be a new category of content that would benefit from its own directory."
        
        # Display the choice
        console.print(Panel(
            f"[yellow]Suggesting NEW directory:[/] {suggested_name}\n"
            f"[blue]Proposed path:[/] {base_dir / suggested_name}\n"
            f"[blue]Reasoning:[/] {reasoning}",
            title="New Directory Suggestion"
        ))
        
        return DirectorySuggestion(base_dir / suggested_name, True)

    def process_batch(self):
        """Process all queued batch requests at once."""
        if not self.batch_queue:
            console.print("[yellow]Batch queue is empty. Nothing to process.[/]")
            return
            
        console.print(f"[cyan]Processing {len(self.batch_queue)} batch requests...[/]")
        
        # Group requests by operation type
        filename_requests = []
        dirname_requests = []
        
        for request in self.batch_queue:
            if request["operation_type"] == "filename":
                filename_requests.append(request)
            elif request["operation_type"] == "dirname":
                dirname_requests.append(request)
        
        # Process filename requests
        if filename_requests:
            self._batch_process_filenames(filename_requests)
            
        # Process dirname requests
        if dirname_requests:
            self._batch_process_dirnames(dirname_requests)
            
        # Clear the queue after processing
        self.batch_queue = []
        
    def _batch_process_filenames(self, requests):
        """Process a batch of filename standardization requests."""
        # Create a batch prompt with all requests
        year_month = datetime.datetime.now().strftime("%Y-%m")
        
        batch_prompt = f"""Given multiple files and descriptions, generate standardized filenames for each.
Format: {self.config["default_filename_pattern"]}

Where:
- {{year}}-{{month}} should be the current date ({year_month}) unless a specific date is mentioned
- {{category}} is a single word representing the domain (e.g., finance, research, project)
- {{descriptor}} is 2-3 words joined by hyphens that capture the main topic
- {{context}} is 1-2 key identifying words

Files to process:
"""

        # Add each file to the prompt
        for i, request in enumerate(requests, 1):
            batch_prompt += f"\nFILE {i}:\nOriginal: {request['file_path'].name}\nDescription: {request['description']}\n"
            
        batch_prompt += "\nReturn results in the following format for each file:\nFILE 1: standardized-filename\nFILE 2: standardized-filename\n...\n"
        
        try:
            response = self._get_ai_completion(batch_prompt)
            
            # Parse the results
            results = {}
            file_num_pattern = re.compile(r"FILE\s+(\d+):\s+(.*)")
            
            for line in response.splitlines():
                match = file_num_pattern.match(line.strip())
                if match:
                    file_num = int(match.group(1))
                    filename = match.group(2).strip()
                    if 1 <= file_num <= len(requests):
                        results[file_num] = filename
            
            # Display results and confirm actions
            console.print("\n[bold cyan]Batch Processing Results:[/]")
            table = Table(title="Filename Suggestions")
            table.add_column("#", style="cyan")
            table.add_column("Original", style="blue")
            table.add_column("Suggested", style="green")
            
            for i, request in enumerate(requests, 1):
                suggested_name = results.get(i, "")
                
                if suggested_name and request["file_path"].suffix and not suggested_name.endswith(request["file_path"].suffix):
                    suggested_name += request["file_path"].suffix
                    
                table.add_row(str(i), request["file_path"].name, suggested_name)
                
            console.print(table)
            
            # Ask which suggestions to apply
            selected = Prompt.ask(
                "\n[bold yellow]Enter the numbers of suggestions to apply (comma-separated, or 'all')[/]",
                default="all"
            )
            
            indices_to_apply = []
            if selected.lower() == "all":
                indices_to_apply = list(range(1, len(requests) + 1))
            else:
                for part in selected.split(","):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part)
                        if 1 <= idx <= len(requests):
                            indices_to_apply.append(idx)
            
            # Apply the selected suggestions
            for idx in indices_to_apply:
                i = idx - 1  # Convert to 0-based index
                request = requests[i]
                suggested_name = results.get(idx, "")
                
                if not suggested_name:
                    console.print(f"[yellow]No valid suggestion for file #{idx}. Skipping.[/]")
                    continue
                    
                # Add file extension if missing
                if request["file_path"].suffix and not suggested_name.endswith(request["file_path"].suffix):
                    suggested_name += request["file_path"].suffix
                
                # Rename the file
                try:
                    new_path = request["file_path"].with_name(suggested_name)
                    request["file_path"].rename(new_path)
                    console.print(f"[green]Renamed: {request['file_path'].name} → {suggested_name}[/]")
                    
                    self._log_operation("rename_file", {
                        "original_path": str(request["file_path"]),
                        "original_name": request["file_path"].name,
                        "new_name": suggested_name,
                        "new_path": str(new_path),
                        "description": request["description"],
                    })
                except Exception as e:
                    console.print(f"[red]Error renaming file #{idx}: {e}[/]")
            
        except Exception as e:
            console.print(f"[red]Error in batch processing: {e}[/]")
            
    def _batch_process_dirnames(self, requests):
        """Process a batch of directory name standardization requests."""
        # Create a batch prompt with all requests
        year_month = datetime.datetime.now().strftime("%Y-%m")
        
        batch_prompt = f"""Given multiple directories and descriptions, generate standardized directory names for each.
Format: {self.config["default_directory_pattern"]}

Where:
- {{category}} is a single word representing the project or domain
- {{purpose}} is a concise description of the directory's purpose

Directories to process:
"""

        # Add each directory to the prompt
        for i, request in enumerate(requests, 1):
            batch_prompt += f"\nDIR {i}:\nOriginal: {request['file_path'].name}\nDescription: {request['description']}\n"
            
        batch_prompt += "\nReturn results in the following format for each directory:\nDIR 1: standardized-dirname\nDIR 2: standardized-dirname\n...\n"
        
        try:
            response = self._get_ai_completion(batch_prompt)
            
            # Parse the results using regex for more robustness
            results = {}
            dir_pattern = re.compile(r"DIR\s+(\d+):\s+(.*)")
            
            for line in response.splitlines():
                match = dir_pattern.match(line.strip())
                if match:
                    dir_num = int(match.group(1))
                    dirname = match.group(2).strip()
                    if 1 <= dir_num <= len(requests):
                        results[dir_num] = dirname
            
            # Display results and confirm actions
            console.print("\n[bold cyan]Batch Processing Results:[/]")
            table = Table(title="Directory Name Suggestions")
            table.add_column("#", style="cyan")
            table.add_column("Original", style="blue")
            table.add_column("Suggested", style="green")
            
            for i, request in enumerate(requests, 1):
                suggested_name = results.get(i, "")
                table.add_row(str(i), request["file_path"].name, suggested_name)
                
            console.print(table)
            
            # Ask which suggestions to apply
            selected = Prompt.ask(
                "\n[bold yellow]Enter the numbers of suggestions to apply (comma-separated, or 'all')[/]",
                default="all"
            )
            
            indices_to_apply = []
            if selected.lower() == "all":
                indices_to_apply = list(range(1, len(requests) + 1))
            else:
                for part in selected.split(","):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part)
                        if 1 <= idx <= len(requests):
                            indices_to_apply.append(idx)
            
            # Apply the selected suggestions
            for idx in indices_to_apply:
                i = idx - 1  # Convert to 0-based index
                request = requests[i]
                suggested_name = results.get(idx, "")
                
                if not suggested_name:
                    console.print(f"[yellow]No valid suggestion for directory #{idx}. Skipping.[/]")
                    continue
                
                # Rename the directory
                try:
                    new_path = request["file_path"].parent / suggested_name
                    request["file_path"].rename(new_path)
                    console.print(f"[green]Renamed: {request['file_path'].name} → {suggested_name}[/]")
                    
                    self._log_operation("rename_directory", {
                        "original_path": str(request["file_path"]),
                        "original_name": request["file_path"].name,
                        "new_name": suggested_name,
                        "new_path": str(new_path),
                        "description": request["description"],
                    })
                    
                    # Register or update directory entry
                    cat_purpose = suggested_name.split("_", 1)
                    category = cat_purpose[0].split("-")[-1] if len(cat_purpose) > 0 else "misc"
                    self._register_directory(new_path, category, request["description"])
                    
                except Exception as e:
                    console.print(f"[red]Error renaming directory #{idx}: {e}[/]")
        except Exception as e:
            console.print(f"[red]Error in batch processing: {e}[/]")

    def process_file(self, file_path: str, batch_mode=False):
        """Process a single file with human-in-the-loop guidance."""
        path = Path(file_path).expanduser().resolve()
        
        if not path.exists():
            console.print(f"[red]Error: File '{file_path}' not found[/]")
            return
        
        if not path.is_file():
            console.print(f"[red]Error: '{file_path}' is not a file[/]")
            return
        
        # Display file info
        metadata = self._extract_file_metadata(path)
        preview = self._generate_file_preview(path)
        
        console.print(Panel(
            f"[bold cyan]File:[/] {path.name}\n"
            f"[cyan]Path:[/] {path}\n"
            f"[cyan]Size:[/] {metadata['size_human']}\n"
            f"[cyan]Modified:[/] {metadata.get('modified', 'Unknown')[:19]}\n\n"
            f"[cyan]Preview:[/]\n{preview[:500]}{'...' if len(preview) > 500 else ''}",
            title="File Analysis"
        ))
        
        # Add option to open the file
        if Confirm.ask("\nWould you like to open this file to review it?"):
            self._open_file(path)
        
        # Get user description
        description = Prompt.ask(
            "\n[bold yellow]Please describe this file's content and purpose[/]\n"
            "[dim](Type 'del' to delete the file, 'open' to open the file, or 'batch' to add to batch queue)[/]"
        )
        
        # Handle open request
        if description.strip().lower() == "open":
            self._open_file(path)
            description = Prompt.ask("[bold yellow]Please describe this file's content and purpose[/]")
        
        # Handle deletion case
        if description.strip().lower() == "del":
            if Confirm.ask(f"[bold red]Are you sure you want to delete {path.name}?[/]"):
                try:
                    path.unlink()
                    console.print(f"[green]File deleted: {path}[/]")
                    self._log_operation("delete_file", {
                        "file_path": str(path),
                        "file_name": path.name,
                    })
                except Exception as e:
                    console.print(f"[red]Error deleting file: {e}[/]")
                return
            else:
                console.print("[yellow]Deletion cancelled.[/]")
                # Continue with normal processing
                description = Prompt.ask("[bold yellow]Please describe this file's content and purpose[/]")
        
        # Handle batch mode
        if description.strip().lower() == "batch":
            console.print("[cyan]Adding file to batch queue for later processing...[/]")
            description = Prompt.ask("[bold yellow]Please provide a description for batch processing[/]")
            self._batch_add_request(path, description, "filename")
            console.print(f"[green]Added {path.name} to batch queue. Current queue size: {len(self.batch_queue)}[/]")
            return
            
        # If already in batch mode, just add to queue and return
        if batch_mode:
            self._batch_add_request(path, description, "filename")
            console.print(f"[green]Added {path.name} to batch queue. Current queue size: {len(self.batch_queue)}[/]")
            return
            
        # Generate standardized filename
        new_name = self._get_standardized_filename(path.name, description)
        
        if not new_name:
            console.print("[red]Failed to generate a standardized filename.[/]")
            return
        
        # Show the suggestion
        console.print(Panel(
            f"[bold cyan]Original:[/] {path.name}\n"
            f"[bold green]Suggested:[/] {new_name}",
            title="Filename Suggestion"
        ))
        
        # Confirm the name
        rename_confirmed = Confirm.ask("Would you like to rename the file?")
        
        if not rename_confirmed:
            # Let the user provide feedback and try again
            feedback = Prompt.ask(
                "[yellow]What would you like to change about the suggested name?[/]\n"
                "[dim](Leave blank to skip renaming)[/]"
            )
            
            if not feedback:
                console.print("[yellow]Skipping rename stage.[/]")
                new_name = path.name  # Keep original name
            else:
                # Try again with feedback
                prompt = f"""Given a file, description, and feedback, generate a revised filename.
    Format: {{year}}-{{month}}-{{category}}_{{descriptor}}_{{context}}

    Original file: {path.name}
    Description: {description}
    Previous suggestion: {new_name}
    User feedback: {feedback}

    Return ONLY the revised filename."""

                try:
                    revised_name = self._get_ai_completion(prompt)
                    
                    # Keep original extension
                    if revised_name and path.suffix and not revised_name.endswith(path.suffix):
                        revised_name += path.suffix
                    
                    console.print(Panel(
                        f"[bold cyan]Original:[/] {path.name}\n"
                        f"[bold yellow]Previous suggestion:[/] {new_name}\n"
                        f"[bold green]Revised suggestion:[/] {revised_name}",
                        title="Revised Filename"
                    ))
                    
                    if Confirm.ask("Use this revised name?"):
                        new_name = revised_name
                    else:
                        console.print("[yellow]Skipping rename stage.[/]")
                        new_name = path.name  # Keep original name
                except Exception as e:
                    console.print(f"[red]Error generating revised name: {e}[/]")
                    new_name = path.name  # Keep original name
        
        # If we're renaming, do it now
        if new_name != path.name:
            try:
                new_path = path.with_name(new_name)
                path.rename(new_path)
                console.print(f"[green]File renamed to: {new_name}[/]")
                
                self._log_operation("rename_file", {
                    "original_path": str(path),
                    "original_name": path.name,
                    "new_name": new_name,
                    "new_path": str(new_path),
                    "description": description,
                })
                
                # Update path for the next steps
                path = new_path
            except Exception as e:
                console.print(f"[red]Error renaming file: {e}[/]")
                # Continue with original path
        
        # Suggest organization
        target_dir, is_new_dir = self._suggest_target_directory(path, new_name, description)
        
        # Ask for confirmation or feedback
        if is_new_dir:
            proceed = Confirm.ask(f"Create new directory '{target_dir.name}' and move file there?")
            
            if not proceed:
                feedback = Prompt.ask(
                    "[yellow]What kind of directory would be better?[/]\n"
                    "[dim](Leave blank to skip moving)[/]"
                )
                
                if not feedback:
                    console.print("[yellow]Skipping organization stage.[/]")
                    return
                
                # Try again with feedback
                prompt = f"""Given a file and feedback, suggest a better directory name.
Format: {{Category}}

Guidelines:
- The directory name should be a simple, clear category (e.g., "Documents", "ProjectFiles", "FinancialReports")
- Use PascalCase (capitalize each word, no spaces)
- Keep the name concise but descriptive (typically 1-2 words)
- The name should be broad enough to group similar files, not overly specific

File: {path.name}
Description: {description}
Previous suggestion: {target_dir.name}
User feedback: {feedback}

Return ONLY the directory name as a single PascalCase word or PascalCase compound word."""

                try:
                    revised_dir_name = self._get_ai_completion(prompt)
                    target_dir = Path(self.config["default_base_dir"]) / revised_dir_name
                    
                    console.print(Panel(
                        f"[bold green]Revised directory suggestion:[/] {target_dir.name}",
                        title="Directory Revision"
                    ))
                    
                    proceed = Confirm.ask(f"Create directory '{target_dir.name}' and move file there?")
                    if not proceed:
                        console.print("[yellow]Skipping organization stage.[/]")
                        return
                except Exception as e:
                    console.print(f"[red]Error generating revised directory: {e}[/]")
                    return
            
            # Create the new directory
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                console.print(f"[green]Created directory: {target_dir}[/]")
                
                # Register the new directory
                cat_purpose = target_dir.name.split("_", 1)
                category = cat_purpose[0].split("-")[-1] if len(cat_purpose) > 0 else "misc"
                self._register_directory(target_dir, category, description)
                
                self._log_operation("create_directory", {
                    "directory_path": str(target_dir),
                    "directory_name": target_dir.name,
                    "description": description,
                })
            except Exception as e:
                console.print(f"[red]Error creating directory: {e}[/]")
                return
        else:
            # Using existing directory
            proceed = True
            if not self.config["skip_confirm_for_move"]:
                proceed = Confirm.ask(f"Move file to '{target_dir}'?")
            
            if not proceed:
                console.print("[yellow]Skipping organization stage.[/]")
                return
        
        # Move the file
        try:
            target_file_path = target_dir / path.name
            shutil.move(str(path), str(target_file_path))
            console.print(f"[green]File moved to: {target_file_path}[/]")
            
            self._log_operation("move_file", {
                "original_path": str(path),
                "new_path": str(target_file_path),
                "target_directory": str(target_dir),
            })
            
        except Exception as e:
            console.print(f"[red]Error moving file: {e}[/]")

    def process_directory(self, directory_path: str):
        """Process a directory with human-in-the-loop guidance."""
        path = Path(directory_path).expanduser().resolve()
        
        if not path.exists():
            console.print(f"[red]Error: Directory '{directory_path}' not found[/]")
            return
        
        if not path.is_dir():
            console.print(f"[red]Error: '{directory_path}' is not a directory[/]")
            return
        
        # Display directory info
        file_count = sum(1 for _ in path.glob("*") if _.is_file())
        subdir_count = sum(1 for _ in path.glob("*") if _.is_dir())
        file_list = list(path.glob("*"))[:10]  # First 10 files/dirs
        
        file_list_str = "\n".join(f"- [dir] {f.name}" if f.is_dir() else f"- {f.name}" for f in file_list)
        if len(list(path.glob("*"))) > 10:
            file_list_str += "\n- ..."
        
        console.print(Panel(
            f"[bold cyan]Directory:[/] {path.name}\n"
            f"[cyan]Path:[/] {path}\n"
            f"[cyan]Contains:[/] {file_count} files, {subdir_count} subdirectories\n\n"
            f"[cyan]Sample Contents:[/]\n{file_list_str}",
            title="Directory Analysis"
        ))
        
        # Add option to open the directory
        if Confirm.ask("\nWould you like to open this directory to explore its contents?"):
            self._open_directory(path)
        
        # Get user description
        description = Prompt.ask(
            "\n[bold yellow]Please describe this directory's content and purpose[/]\n"
            "[dim](Type 'del' to delete the directory, 'open' to open in file explorer)[/]"
        )
        
        # Handle open request
        if description.strip().lower() == "open":
            self._open_directory(path)
            description = Prompt.ask("[bold yellow]Please describe this directory's content and purpose[/]")
        
        # Handle deletion case
        if description.strip().lower() == "del":
            if Confirm.ask(f"[bold red]Are you sure you want to delete {path.name} and ALL its contents?[/]"):
                try:
                    shutil.rmtree(path)
                    console.print(f"[green]Directory deleted: {path}[/]")
                    self._log_operation("delete_directory", {
                        "directory_path": str(path),
                        "directory_name": path.name,
                    })
                except Exception as e:
                    console.print(f"[red]Error deleting directory: {e}[/]")
                return
            else:
                console.print("[yellow]Deletion cancelled.[/]")
                # Continue with normal processing
                description = Prompt.ask("[bold yellow]Please describe this directory's content and purpose[/]")
        
        # Generate standardized directory name
        new_name = self._get_standardized_dirname(path, description)
        
        if not new_name:
            console.print("[red]Failed to generate a standardized directory name.[/]")
            return
        
        # Show the suggestion
        console.print(Panel(
            f"[bold cyan]Original:[/] {path.name}\n"
            f"[bold green]Suggested:[/] {new_name}",
            title="Directory Name Suggestion"
        ))
        
        # Confirm the name
        rename_confirmed = Confirm.ask("Would you like to rename the directory?")
        
        if not rename_confirmed:
            # Let the user provide feedback and try again
            feedback = Prompt.ask(
                "[yellow]What would you like to change about the suggested name?[/]\n"
                "[dim](Leave blank to skip renaming)[/]"
            )
            
            if not feedback:
                console.print("[yellow]Skipping rename stage.[/]")
                new_name = path.name  # Keep original name
            else:
                # Try again with feedback
                prompt = f"""Given a directory, description, and feedback, generate a revised directory name.
    Format: {{year}}-{{month}}-{{category}}_{{purpose}}

    Original directory: {path.name}
    Description: {description}
    Previous suggestion: {new_name}
    User feedback: {feedback}

    Return ONLY the revised directory name."""

                try:
                    revised_name = self._get_ai_completion(prompt)
                    
                    console.print(Panel(
                        f"[bold cyan]Original:[/] {path.name}\n"
                        f"[bold yellow]Previous suggestion:[/] {new_name}\n"
                        f"[bold green]Revised suggestion:[/] {revised_name}",
                        title="Revised Directory Name"
                    ))
                    
                    if Confirm.ask("Use this revised name?"):
                        new_name = revised_name
                    else:
                        console.print("[yellow]Skipping rename stage.[/]")
                        new_name = path.name  # Keep original name
                except Exception as e:
                    console.print(f"[red]Error generating revised name: {e}[/]")
                    new_name = path.name  # Keep original name
        
        # If we're renaming, do it now
        if new_name != path.name:
            try:
                new_path = path.parent / new_name
                path.rename(new_path)
                console.print(f"[green]Directory renamed to: {new_name}[/]")
                
                self._log_operation("rename_directory", {
                    "original_path": str(path),
                    "original_name": path.name,
                    "new_name": new_name,
                    "new_path": str(new_path),
                    "description": description,
                })
                
                # Register or update directory entry
                cat_purpose = new_name.split("_", 1)
                category = cat_purpose[0].split("-")[-1] if len(cat_purpose) > 0 else "misc"
                purpose = cat_purpose[1] if len(cat_purpose) > 1 else "general"
                
                self._register_directory(new_path, category, description)
                
                # Update path for the next steps
                path = new_path
            except Exception as e:
                console.print(f"[red]Error renaming directory: {e}[/]")
                # Continue with original path
        
        # Ask if user wants to process files within the directory
        if Confirm.ask("Would you like to process files within this directory?"):
            file_paths = [f for f in path.glob("*") if f.is_file()]
            
            if not file_paths:
                console.print("[yellow]No files found in this directory.[/]")
                return
            
            console.print(f"[cyan]Found {len(file_paths)} files in the directory.[/]")
            
            # Ask if user wants to process all files with the same description
            batch_process = Confirm.ask("Process all files with the same description?")
            
            if batch_process:
                batch_description = Prompt.ask("[bold yellow]Please provide a description for all files[/]")
                
                for file_path in file_paths:
                    console.print(f"\n[bold]Processing:[/] {file_path.name}")
                    
                    # Generate standardized filename
                    file_new_name = self._get_standardized_filename(file_path.name, batch_description)
                    
                    if not file_new_name:
                        console.print(f"[red]Failed to generate standardized name for {file_path.name}. Skipping.[/]")
                        continue
                    
                    # Rename file
                    try:
                        file_new_path = file_path.with_name(file_new_name)
                        file_path.rename(file_new_path)
                        console.print(f"[green]File renamed to: {file_new_name}[/]")
                        
                        self._log_operation("rename_file", {
                            "original_path": str(file_path),
                            "original_name": file_path.name,
                            "new_name": file_new_name,
                            "new_path": str(file_new_path),
                            "description": batch_description,
                        })
                    except Exception as e:
                        console.print(f"[red]Error renaming file: {e}[/]")
            else:
                # Process files individually
                for file_path in file_paths:
                    if Confirm.ask(f"\nProcess file '{file_path.name}'?"):
                        self.process_file(str(file_path))
        
        console.print(f"[bold green]Directory processing complete: {path}[/]")

    def view_operation_log(self):
        """Display the operation log to the user."""
        try:
            if not LOG_FILE.exists():
                console.print("[yellow]No operation log found.[/]")
                return
            
            with open(LOG_FILE, "r") as f:
                log_entries = [json.loads(line) for line in f if line.strip()]
            
            if not log_entries:
                console.print("[yellow]Operation log is empty.[/]")
                return
            
            # Create a table to display the log
            table = Table(title="Operation Log")
            table.add_column("Time", style="cyan")
            table.add_column("Operation", style="green")
            table.add_column("Details", style="white")
            
            # Add the most recent entries first (up to 20)
            for entry in reversed(log_entries[-20:]):
                timestamp = entry.get("timestamp", "")[:19]  # Truncate to remove milliseconds
                operation = entry.get("operation", "")
                
                # Format details differently based on operation type
                details = ""
                if operation == "rename_file":
                    details = f"'{entry.get('original_name', '')}' → '{entry.get('new_name', '')}'"
                elif operation == "move_file":
                    details = f"Moved to {Path(entry.get('target_directory', '')).name}"
                elif operation == "delete_file":
                    details = f"Deleted '{entry.get('file_name', '')}'"
                elif operation == "rename_directory":
                    details = f"'{entry.get('original_name', '')}' → '{entry.get('new_name', '')}'"
                elif operation == "create_directory":
                    details = f"Created '{entry.get('directory_name', '')}'"
                elif operation == "delete_directory":
                    details = f"Deleted '{entry.get('directory_name', '')}'"
                
                table.add_row(timestamp, operation, details)
            
            console.print(table)
            
            # Offer to export the log
            if Confirm.ask("Would you like to export the full log to a CSV file?"):
                export_path = Path.home() / "smart_organizer_log.csv"
                with open(export_path, "w") as f:
                    f.write("Timestamp,Operation,Original Name,New Name,Path\n")
                    for entry in log_entries:
                        timestamp = entry.get("timestamp", "")[:19]
                        operation = entry.get("operation", "")
                        original_name = entry.get("original_name", "")
                        new_name = entry.get("new_name", "")
                        path = entry.get("new_path", entry.get("original_path", ""))
                        f.write(f'"{timestamp}","{operation}","{original_name}","{new_name}","{path}"\n')
                
                console.print(f"[green]Log exported to: {export_path}[/]")
        
        except Exception as e:
            console.print(f"[red]Error displaying operation log: {e}[/]")

    def _batch_processing_mode(self):
        """Enter batch processing mode."""
        console.print(Panel(
            "Batch Processing Mode\n"
            "Add multiple files to the queue for bulk processing.",
            title="Batch Mode"
        ))
        
        # Display current queue
        if self.batch_queue:
            console.print(f"[cyan]Current queue has {len(self.batch_queue)} items.[/]")
            if Confirm.ask("View current queue?"):
                table = Table(title="Batch Queue")
                table.add_column("#", style="cyan")
                table.add_column("Path", style="blue")
                table.add_column("Type", style="green")
                
                for i, item in enumerate(self.batch_queue, 1):
                    table.add_row(
                        str(i),
                        str(item["file_path"]),
                        item["operation_type"]
                    )
                
                console.print(table)
        
        # Batch actions menu
        while True:
            console.print("\n[bold cyan]Batch Actions:[/]")
            console.print("1. Add files to batch")
            console.print("2. Add directory to batch")
            console.print("3. Process batch queue")
            console.print("4. Clear batch queue")
            console.print("5. Return to main menu")
            
            choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4", "5"])
            
            if choice == "1":
                file_path = Prompt.ask("[bold]Enter the path to a file[/]")
                path = Path(file_path).expanduser().resolve()
                
                if not path.exists() or not path.is_file():
                    console.print(f"[red]Error: '{file_path}' is not a valid file[/]")
                    continue
                
                description = Prompt.ask("[bold]Enter a description for this file[/]")
                self._batch_add_request(path, description, "filename")
                console.print(f"[green]Added to batch queue. Queue size: {len(self.batch_queue)}[/]")
                
            elif choice == "2":
                dir_path = Prompt.ask("[bold]Enter the path to a directory[/]")
                path = Path(dir_path).expanduser().resolve()
                
                if not path.exists() or not path.is_dir():
                    console.print(f"[red]Error: '{dir_path}' is not a valid directory[/]")
                    continue
                
                description = Prompt.ask("[bold]Enter a description for this directory[/]")
                self._batch_add_request(path, description, "dirname")
                console.print(f"[green]Added to batch queue. Queue size: {len(self.batch_queue)}[/]")
                
            elif choice == "3":
                if not self.batch_queue:
                    console.print("[yellow]Batch queue is empty. Nothing to process.[/]")
                else:
                    self.process_batch()
                
            elif choice == "4":
                if Confirm.ask("[bold red]Are you sure you want to clear the batch queue?[/]"):
                    self.batch_queue = []
                    console.print("[green]Batch queue cleared.[/]")
                
            elif choice == "5":
                break

    def manage_directories(self):
        """Manage the directory registry."""
        while True:
            directories = self._load_directory_registry()
            
            # Display the current directories
            table = Table(title="Registered Directories")
            table.add_column("#", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Category", style="blue")
            table.add_column("Description", style="white")
            
            for i, directory in enumerate(directories, 1):
                table.add_row(
                    str(i),
                    directory.get("name", ""),
                    directory.get("category", ""),
                    directory.get("description", "")
                )
            
            console.print(table)
            
            # Display options
            console.print("\n[bold cyan]Directory Management Options:[/]")
            console.print("1. Add a new directory")
            console.print("2. Remove a directory")
            console.print("3. Update directory information")
            console.print("4. Return to main menu")
            
            choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4"])
            
            if choice == "1":
                # Add a new directory
                path_str = Prompt.ask("[bold]Enter the directory path[/]")
                path = Path(path_str).expanduser().resolve()
                
                if not path.exists():
                    if Confirm.ask(f"Directory '{path}' doesn't exist. Create it?"):
                        try:
                            path.mkdir(parents=True)
                        except Exception as e:
                            console.print(f"[red]Error creating directory: {e}[/]")
                            continue
                    else:
                        continue
                
                category = Prompt.ask("[bold]Enter a category for this directory[/]")
                description = Prompt.ask("[bold]Enter a description[/]")
                
                self._register_directory(path, category, description)
                console.print(f"[green]Directory '{path}' registered.[/]")
            
            elif choice == "2":
                # Remove a directory
                if not directories:
                    console.print("[yellow]No directories to remove.[/]")
                    continue
                
                idx = Prompt.ask(
                    "[bold]Enter the number of the directory to remove[/]",
                    choices=[str(i) for i in range(1, len(directories) + 1)]
                )
                
                idx = int(idx) - 1
                removed = directories.pop(idx)
                self._save_directory_registry(directories)
                console.print(f"[green]Removed directory '{removed['name']}' from registry.[/]")
            
            elif choice == "3":
                # Update directory information
                if not directories:
                    console.print("[yellow]No directories to update.[/]")
                    continue
                
                idx = Prompt.ask(
                    "[bold]Enter the number of the directory to update[/]",
                    choices=[str(i) for i in range(1, len(directories) + 1)]
                )
                
                idx = int(idx) - 1
                dir_to_update = directories[idx]
                
                console.print(f"[bold]Updating: {dir_to_update['name']}[/]")
                
                new_category = Prompt.ask(
                    "[bold]Enter new category[/]",
                    default=dir_to_update.get("category", "")
                )
                
                new_description = Prompt.ask(
                    "[bold]Enter new description[/]",
                    default=dir_to_update.get("description", "")
                )
                
                dir_to_update["category"] = new_category
                dir_to_update["description"] = new_description
                dir_to_update["last_updated"] = datetime.datetime.now().isoformat()
                
                self._save_directory_registry(directories)
                console.print(f"[green]Updated directory '{dir_to_update['name']}'.[/]")
            
            else:
                # Return to main menu
                break

    def _edit_configuration(self):
        """Edit the application configuration."""
        console.print(Panel(
            "Configuration Editor\n"
            "Modify the behavior of Smart Organizer.",
            title="Configuration"
        ))
        
        # Display current configuration
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in self.config.items():
            table.add_row(key, str(value))
        
        console.print(table)
        
        # Edit configuration
        if Confirm.ask("\nWould you like to edit the configuration?"):
            setting = Prompt.ask(
                "[bold]Enter the setting to edit[/]",
                choices=list(self.config.keys())
            )
            
            current_value = self.config[setting]
            new_value = Prompt.ask(
                f"[bold]Enter new value for {setting}[/]",
                default=str(current_value)
            )
            
            # Handle different value types
            if isinstance(current_value, bool):
                new_value = new_value.lower() in ["true", "yes", "y", "1"]
            elif isinstance(current_value, int):
                try:
                    new_value = int(new_value)
                except ValueError:
                    console.print("[red]Invalid integer value. Setting not changed.[/]")
                    return
            elif isinstance(current_value, float):
                try:
                    new_value = float(new_value)
                except ValueError:
                    console.print("[red]Invalid float value. Setting not changed.[/]")
                    return
            
            # Update configuration
            self.config[setting] = new_value
            
            # Save to file
            try:
                with open(CONFIG_PATH, "w") as f:
                    json.dump(self.config, f, indent=2)
                console.print(f"[green]Configuration updated: {setting} = {new_value}[/]")
            except Exception as e:
                console.print(f"[red]Error saving configuration: {e}[/]")

    def run(self):
        """Run the main application loop."""
        console.print(Panel(
            "Welcome to Smart Organizer - AI-powered file management utility",
            style="bold green"
        ))
        
        while True:
            console.print("\n[bold cyan]Main Menu:[/]")
            console.print("1. Process a file")
            console.print("2. Process a directory")
            console.print("3. Batch processing mode")
            console.print("4. Manage registered directories")
            console.print("5. View operation log")
            console.print("6. Configuration")
            console.print("7. Exit")
            
            choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4", "5", "6", "7"])
            
            if choice == "1":
                file_path = Prompt.ask("[bold]Enter the path to a file[/]")
                self.process_file(file_path)
                
            elif choice == "2":
                dir_path = Prompt.ask("[bold]Enter the path to a directory[/]")
                self.process_directory(dir_path)
                
            elif choice == "3":
                self._batch_processing_mode()
                
            elif choice == "4":
                self.manage_directories()
                
            elif choice == "5":
                self.view_operation_log()
                
            elif choice == "6":
                self._edit_configuration()
                
            elif choice == "7":
                console.print("[green]Thank you for using Smart Organizer![/]")
                break


# Add the command-line interface functions
def display_help():
    """Display help information."""
    help_text = """
# Smart Organizer

A human-in-the-loop AI command line tool for file and directory management.

## Commands

- `smart_organizer file <path>` - Process a single file
- `smart_organizer dir <path>` - Process a directory
- `smart_organizer batch` - Process files in batch mode
- `smart_organizer log` - View operation log
- `smart_organizer dirs` - Manage registered directories
- `smart_organizer config` - Configure settings
- `smart_organizer help` - Display this help message

## Features

- AI-powered filename standardization
- Intelligent directory organization
- Batch processing for efficiency
- Operation logging
- File deletion with confirmation
    """
    
    console.print(Markdown(help_text))

def configure_settings():
    """Configure tool settings."""
    try:
        # Load current config
        if not CONFIG_PATH.exists():
            config = DEFAULT_CONFIG
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
        else:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
        
        # Display current settings
        console.print("\n[bold cyan]Current Settings:[/]")
        for key, value in config.items():
            console.print(f"[blue]{key}:[/] {value}")
        
        # Ask which setting to change
        console.print("\n[bold cyan]Which setting would you like to change?[/]")
        for i, key in enumerate(config.keys(), 1):
            console.print(f"{i}. {key}")
        console.print(f"{len(config) + 1}. Save and return")
        
        choice = Prompt.ask(
            "\nEnter your choice",
            choices=[str(i) for i in range(1, len(config) + 2)]
        )
        
        if int(choice) <= len(config):
            # Update the selected setting
            setting_key = list(config.keys())[int(choice) - 1]
            
            # Handle different setting types
            if isinstance(config[setting_key], bool):
                config[setting_key] = Confirm.ask(f"Enable {setting_key}?", default=config[setting_key])
            elif isinstance(config[setting_key], int):
                config[setting_key] = int(Prompt.ask(
                    f"Enter new value for {setting_key}",
                    default=str(config[setting_key])
                ))
            elif isinstance(config[setting_key], float):
                config[setting_key] = float(Prompt.ask(
                    f"Enter new value for {setting_key}",
                    default=str(config[setting_key])
                ))
            else:
                config[setting_key] = Prompt.ask(
                    f"Enter new value for {setting_key}",
                    default=str(config[setting_key])
                )
            
            # Save the updated config
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
            
            console.print(f"[green]Updated {setting_key} to {config[setting_key]}[/]")
            
            # Ask if they want to update more settings
            if Confirm.ask("Would you like to update more settings?"):
                configure_settings()
        
    except Exception as e:
        console.print(f"[red]Error configuring settings: {e}[/]")

def main():
    """Main function to run the Smart Organizer tool."""
    if len(sys.argv) < 2:
        # Run in interactive mode
        organizer = SmartOrganizer()
        organizer.run()
        return
    
    command = sys.argv[1].lower()
    
    if command == "help":
        display_help()
        return
    
    organizer = SmartOrganizer()
    
    if command == "file" and len(sys.argv) > 2:
        organizer.process_file(sys.argv[2])
    
    elif command == "dir" and len(sys.argv) > 2:
        organizer.process_directory(sys.argv[2])
        
    elif command == "batch":
        organizer._batch_processing_mode()
    
    elif command == "log":
        organizer.view_operation_log()
    
    elif command == "dirs":
        organizer.manage_directories()
    
    elif command == "config":
        configure_settings()
    
    else:
        console.print("[red]Invalid command. Use 'smart_organizer help' for usage information.[/]")

if __name__ == "__main__":
    main()
