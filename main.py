from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any, List
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import subprocess
import glob
import httpx
from pydantic import BaseModel
from enum import Enum
import logging
import re
import base64
from PIL import Image
import io
import markdown
import csv
import duckdb
import git
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.getcwd(), 'data'))
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDIxMjFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.QZctdKY8bEGBPfclmVyLmngciYa4SoAvWPa1vA3PYJM"
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable must be set")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

class TaskStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"

class TaskResult(BaseModel):
    status: TaskStatus
    message: str
    output: Optional[str] = None

def resolve_path(file_path: str) -> str:
    """
    Resolve and validate a file path.
    Ensures paths are within DATA_DIR for security.
    """
    # Remove /data/ prefix if present
    if file_path.startswith('/data/'):
        file_path = file_path[6:]
    
    # Get absolute path
    full_path = os.path.abspath(os.path.join(DATA_DIR, file_path))
    
    # Security check: ensure path is within DATA_DIR
    if not full_path.startswith(os.path.abspath(DATA_DIR)):
        raise ValueError("Access denied: Path must be within /data directory")
        
    return full_path

class TaskProcessor:
    def __init__(self):
        self.llm_headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
        
    async def process_task(self, task_description: str) -> TaskResult:
        """Process the given task and return the result."""
        try:
            logger.info(f"Processing task: {task_description}")
            
            # Use LLM to parse task
            parse_prompt = f"""Parse this task into components:
            - Identify the task type (e.g., format, count, sort)
            - Extract input and output file paths
            - Extract any parameters or requirements
            
            Task: {task_description}"""
            
            parsed = await self._call_llm(parse_prompt)
            task_components = self._interpret_llm_response(parsed)
            
            output = await self._execute_steps(task_components)
            
            return TaskResult(
                status=TaskStatus.SUCCESS,
                message="Task completed successfully",
                output=output
            )
            
        except ValueError as e:
            logger.error(f"Task processing error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"System error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal processing error")

    async def _call_llm(self, prompt: str) -> str:
        """Call GPT-4o-Mini via AI Proxy."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://api.aiproxy.pro/v1/chat/completions",
                headers=self.llm_headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            return response.json()["choices"][0]["message"]["content"]

    def _interpret_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Convert LLM's task parsing response into structured components."""
        # This is a simplified interpretation - in practice, you'd want more robust parsing
        components = {
            "action": None,
            "input_path": None,
            "output_path": None,
            "parameters": {}
        }
        
        # Basic parsing - you'd want to improve this based on actual LLM responses
        if "format" in llm_response.lower() and "prettier" in llm_response.lower():
            components["action"] = "format_prettier"
        elif "wednesday" in llm_response.lower() or "count" in llm_response.lower():
            components["action"] = "count_days"
        elif "sort" in llm_response.lower() and "contact" in llm_response.lower():
            components["action"] = "sort_contacts"
        elif "log" in llm_response.lower() and "recent" in llm_response.lower():
            components["action"] = "recent_logs"
        elif "markdown" in llm_response.lower() and "index" in llm_response.lower():
            components["action"] = "create_index"
        # Add more action types as needed
        
        # Extract paths
        paths = re.findall(r'/data/[^\s]+', llm_response)
        if paths:
            components["input_path"] = paths[0]
            if len(paths) > 1:
                components["output_path"] = paths[1]
                
        return components

    async def _execute_steps(self, task: Dict[str, Any]) -> str:
        """Execute the task steps based on the parsed components."""
        action = task["action"]
        input_path = task.get("input_path")
        output_path = task.get("output_path")
        
        if input_path:
            input_path = resolve_path(input_path)
        if output_path:
            output_path = resolve_path(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Phase A tasks
            if action == "format_prettier":
                return await self._format_prettier(input_path)
            elif action == "count_days":
                return await self._count_days(input_path, output_path)
            elif action == "sort_contacts":
                return await self._sort_contacts(input_path, output_path)
            elif action == "recent_logs":
                return await self._recent_logs(input_path, output_path)
            elif action == "create_index":
                return await self._create_markdown_index(input_path, output_path)
            # Add more task handlers
            
            raise ValueError(f"Unsupported action: {action}")
            
        except Exception as e:
            raise ValueError(f"Error executing task: {str(e)}")

    # Task implementation methods
    async def _format_prettier(self, file_path: str) -> str:
        """Format a file using prettier."""
        try:
            subprocess.run([
                'npx',
                'prettier@3.4.2',
                '--write',
                file_path
            ], check=True)
            return f"Formatted {file_path} with prettier@3.4.2"
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Prettier formatting failed: {str(e)}")

    async def _count_days(self, input_path: str, output_path: str) -> str:
        """Count specific days in a dates file."""
        with open(input_path, 'r') as f:
            dates = f.readlines()
        
        count = sum(
            1 for date_str in dates
            if datetime.strptime(date_str.strip(), '%Y-%m-%d').weekday() == 2
        )
        
        with open(output_path, 'w') as f:
            f.write(str(count))
        
        return f"Counted {count} occurrences"

    async def _sort_contacts(self, input_path: str, output_path: str) -> str:
        """Sort contacts by last_name, first_name."""
        with open(input_path, 'r') as f:
            contacts = json.load(f)
        
        sorted_contacts = sorted(
            contacts,
            key=lambda x: (x['last_name'], x['first_name'])
        )
        
        with open(output_path, 'w') as f:
            json.dump(sorted_contacts, f, indent=2)
        
        return "Sorted contacts written to file"

    async def _recent_logs(self, input_dir: str, output_path: str) -> str:
        """Get first lines of recent log files."""
        log_files = sorted(
            glob.glob(os.path.join(input_dir, '*.log')),
            key=os.path.getmtime,
            reverse=True
        )[:10]
        
        first_lines = []
        for log_file in log_files:
            with open(log_file, 'r') as f:
                first_lines.append(f.readline().strip())
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(first_lines))
        
        return "Wrote first lines of recent logs"

    async def _create_markdown_index(self, input_dir: str, output_path: str) -> str:
        """Create index of markdown H1 headers."""
        index = {}
        for md_file in glob.glob(os.path.join(input_dir, '**/*.md'), recursive=True):
            with open(md_file, 'r') as f:
                for line in f:
                    if line.startswith('# '):
                        relative_path = os.path.relpath(md_file, input_dir)
                        index[relative_path] = line[2:].strip()
                        break
        
        with open(output_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        return "Created markdown index"

# Initialize FastAPI app
app = FastAPI(title="DataWorks Task Processing API")
processor = TaskProcessor()

@app.post("/run")
async def run_task(task: str):
    """
    Execute a task based on plain-English description.
    Returns the task result or appropriate error response.
    """
    try:
        result = await processor.process_task(task)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/read")
async def read_file(path: str):
    """
    Read and return the content of the specified file.
    Returns 404 if file not found.
    """
    try:
        full_path = resolve_path(path)
        logger.info(f"Reading file: {full_path}")
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
            
        with open(full_path, "r") as f:
            content = f.read()
        return content
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error reading file")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)