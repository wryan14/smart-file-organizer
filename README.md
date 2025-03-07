# Smart Organizer

Smart Organizer is a human-in-the-loop AI command line tool for file and directory management. It helps you standardize filenames, organize your files into appropriate directories, and maintain a clean, well-structured file system.

## Features

- ✨ **AI-Powered Naming**: Standardize filenames and directory names with consistent patterns
- 📁 **Intelligent Organization**: Suggest appropriate directories for files based on content
- 🔄 **Human-in-the-Loop**: You approve and refine all suggestions
- 🚀 **Batch Processing**: Process multiple files at once for efficiency
- 📋 **Operation Logging**: Keep track of all file operations
- 🗑️ **Quick Deletion**: Simple "del" keyword to delete files or directories
- 🔍 **Context-Aware**: Uses file content and description to make intelligent decisions

## Installation

### Windows

1. Download the files in this package
2. Run the PowerShell script as administrator:
   ```
   powershell -ExecutionPolicy Bypass -File Install-SmartOrganizer.ps1
   ```
3. Follow the prompts to complete installation
4. Desktop shortcuts will be created automatically

### Manual Installation (All Platforms)

1. Ensure Python 3.6+ is installed
2. Install required packages:
   ```
   pip install rich openai python-dotenv
   ```
3. Set your OpenAI API key as an environment variable or in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the script directly:
   ```
   python smart_organizer.py
   ```

## Usage

### Processing Files

The tool helps you:
1. Standardize filenames using a consistent pattern
2. Move files to appropriate directories
3. Maintain a log of all operations

To process a file:
```
python smart_organizer.py file /path/to/your/file
```

Or use the desktop shortcut or batch file on Windows.

### Processing Directories

The tool helps you:
1. Standardize directory names using a consistent pattern
2. Process multiple files within a directory
3. Maintain organization of nested content

To process a directory:
```
python smart_organizer.py dir /path/to/your/directory
```

Or use the desktop shortcut or batch file on Windows.

### Deleting Files or Directories

To quickly delete a file or directory:
1. When prompted for a description, simply type `del`
2. Confirm the deletion

### Command Line Reference

```
smart_organizer file <path>     Process a single file
smart_organizer dir <path>      Process a directory
smart_organizer batch           Process files in batch mode
smart_organizer log             View operation log
smart_organizer dirs            Manage registered directories
smart_organizer config          Configure settings
smart_organizer help            Display help information
```

## Batch Processing

Smart Organizer supports batch processing to improve efficiency:

1. Run the batch processor:
   ```
   python smart_organizer.py batch
   ```

2. Add files to the queue:
   - Enter file paths one by one
   - Provide a brief description for each
   - Type 'done' when finished adding files

3. Process all files at once:
   - The AI will generate names for all files in a single request
   - Review and select which suggestions to apply
   - Apply changes in bulk

This dramatically speeds up the workflow by reducing the number of API calls and waiting time.

## Naming Patterns

By default, Smart Organizer uses these naming patterns:

- **Files**: `{year}-{month}-{category}_{descriptor}_{context}.ext`
- **Directories**: `{year}-{month}-{category}_{purpose}`

For example:
- `2023-06-finance_quarterly-report_q2.xlsx`
- `2023-06-project_website-redesign`

These patterns can be customized in the configuration.

## Directory Registry

Smart Organizer maintains a registry of known directories to help with organization. You can:

- View all registere
