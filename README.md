# Setup and Execution Guide

## Virtual Environment Setup

This project uses Python virtual environments (`venv`) to manage dependencies.

### Creating and Activating the Virtual Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

### Installing Dependencies

Once the virtual environment is activated, install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Project

After installing dependencies, execute the project:

```bash
python main.py
```

(Replace `main.py` with the appropriate entry point for this project)

### Deactivating the Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

## Requirements File

The `requirements.txt` file contains all project dependencies. To update it after installing new packages:

```bash
pip freeze > requirements.txt
```