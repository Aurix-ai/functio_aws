from typing import Dict, Any
from pathlib import Path
import json
import re
import logging

def read_file(file_path: Path) -> str | None:
    """Reads a file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def load_json(file_path: Path) -> dict | None:
    """Loads a JSON file."""
    content = read_file(file_path)
    if content:
        return json.loads(content)
    return None
    
class PromptConstructor:
    def __init__(self, prompt_templates: Dict[str, str], prompt_context: Dict[str, Any]):
        self.prompt_templates = prompt_templates
        self.prompt_context = prompt_context

    @classmethod
    def create(cls, templates_dir: Path = Path("llm_guides")) -> 'PromptConstructor':
        prompt_templates = cls._load_all_templates(templates_dir)
        prompt_context = cls._load_context(templates_dir)
        return cls(prompt_templates=prompt_templates, prompt_context=prompt_context)
    
    @staticmethod
    def _load_all_templates(templates_dir: Path) -> Dict[str, str]:
        prompt_templates = {}
        for template_file in templates_dir.glob("*.txt"):
            prompt_templates[template_file.stem] = template_file.read_text()
        return prompt_templates
    
    @staticmethod
    def _load_context(templates_dir: Path) -> Dict[str, Any]:
        context_file = templates_dir / "context.json"
        return load_json(context_file)
    
    def construct(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.prompt_templates.get(template_name)
        placeholders = re.findall(r"\{\{([A-Z_]+)\}\}", template)
        for placeholder in placeholders:
            value = context.get(placeholder)
            if value is None:
                logging.warning(f"Placeholder '{{{{{placeholder}}}}}' found in template '{template_name}' but not provided in context. Replacing with empty string.")
                value = ""
            template = template.replace(f"{{{{{placeholder}}}}}", str(value))
        return template