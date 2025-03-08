# Smart Organizer

> AI-powered file organization with a human touch

Smart Organizer is an intelligent command-line tool that helps you bring order to your digital files through AI-assisted naming and organization, while keeping you in control of the process.



## 🌟 Key Features

- **AI-Powered Naming**: Generate consistent, meaningful filenames based on content and purpose
- **Intelligent Directory Suggestions**: Get smart recommendations for where files should live
- **Human-in-the-Loop Control**: You approve and refine all suggestions before changes happen
- **Batch Processing**: Handle multiple files efficiently in a single session
- **Operation Logging**: Keep track of all file management activities
- **Interactive CLI**: Beautiful command-line interface with rich formatting

## 🚀 Quick Start

### Prerequisites

- Python 3.6+
- OpenAI API key

### Installation

**Windows (Automated)**:
```powershell
powershell -ExecutionPolicy Bypass -File Install-SmartOrganizer.ps1
```

**Manual (All Platforms)**:
```bash
# 1. Install required packages
pip install rich openai python-dotenv

# 2. Set your OpenAI API key in a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 3. Run the script
python smart_organizer.py
```

## 💡 Common Workflows

### Workflow 1: Organizing Individual Files

Ideal for handling important documents that need careful organization:

```bash
python smart_organizer.py file /path/to/quarterly_report.pdf
```

1. The tool shows you a preview of the file content
2. You provide a brief description (e.g., "Q2 financial report with performance metrics")
3. AI suggests a standardized name (e.g., "2025-03-finance_quarterly-report_q2.pdf")
4. AI recommends an appropriate directory based on your existing structure
5. You approve or modify each suggestion
6. The file is renamed and moved

### Workflow 2: Cleaning Up a Directory

Perfect for organizing folders with mixed content:

```bash
python smart_organizer.py dir /path/to/messy_downloads
```

1. The tool analyzes the directory content
2. You provide a description of the directory's purpose
3. AI suggests a standardized name for the directory
4. You can choose to process all files within the directory:
   - With the same description (batch mode)
   - Individually with custom descriptions

### Workflow 3: Batch Processing Similar Files

Efficient for handling multiple related files:

```bash
python smart_organizer.py batch
```

1. Add multiple files to the processing queue
2. Provide descriptions for each
3. Process all files at once
4. Review AI suggestions for each file
5. Apply changes in bulk

## 📋 File Naming Patterns

Smart Organizer uses consistent naming patterns:

**Files**: `{year}-{month}-{category}_{descriptor}_{context}.ext`
- Example: `2025-03-project_website-redesign_homepage.psd`

**Directories**: `{year}-{month}-{category}_{purpose}`
- Example: `2025-03-marketing_social-media-campaign`

*These patterns can be customized in the configuration.*

## 🔍 Command Reference

```
smart_organizer file <path>     Process a single file
smart_organizer dir <path>      Process a directory
smart_organizer batch           Batch processing mode
smart_organizer log             View operation log
smart_organizer dirs            Manage directory registry
smart_organizer config          Configure settings
smart_organizer help            Display help
```

## ⚙️ Advanced Usage

### Managing Your Directory Registry

Smart Organizer maintains a registry of known directories to help with organization:

```bash
python smart_organizer.py dirs
```

This allows you to:
- View all registered directories
- Add new directories to the registry
- Update directory information
- Remove directories from the registry

### Quick Deletion

To quickly delete a file or directory:
1. When prompted for a description, type `del`
2. Confirm the deletion

### Customizing Configuration

Adjust settings to match your workflow:

```bash
python smart_organizer.py config
```

Configurable options include:
- Base directory for new folders
- Filename and directory patterns
- AI model selection
- Confidence thresholds

## 📝 Tips for Effective Use

1. **Be Descriptive**: Provide detailed descriptions for better AI suggestions
2. **Use Batch Mode**: Process similar files together for efficiency
3. **Maintain Your Registry**: Keep your directory registry well-organized
4. **Review the Log**: Check the operation log to track changes
5. **Customize Patterns**: Adjust the default patterns to match your organization style

## 🔧 Troubleshooting

**API Key Issues**
- Ensure your OpenAI API key is correctly set in your `.env` file
- Verify you have sufficient API credits

**File Permission Errors**
- Check that you have write permissions for the directories
- Run as administrator if needed for system files

**Improving AI Suggestions**
- Provide more detailed descriptions
- Adjust the AI model in configuration settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
