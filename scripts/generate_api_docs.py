#!/usr/bin/env python
"""Script to automatically generate API documentation from docstrings."""
import inspect
import os
from importlib import import_module
from pathlib import Path


def generate_module_docs(module_path: str, output_dir: Path) -> None:
    """Generate documentation for a Python module."""
    try:
        # Convert path to module name (e.g., src/api/routes.py -> src.api.routes)
        module_name = module_path.replace("/", ".").replace(".py", "")
        module = import_module(module_name)

        # Create markdown content
        content = [f"# {module_name.split('.')[-1].title()} Module\n"]
        content.append(module.__doc__ or "")

        # Document classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                content.append(f"\n## {name}\n")
                content.append(obj.__doc__ or "")

                # Document methods
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if not method_name.startswith("_"):
                        content.append(f"\n### {method_name}\n")
                        content.append(method.__doc__ or "")

        # Write to file
        output_file = output_dir / f"{module_name.split('.')[-1]}.md"
        output_file.write_text("\n".join(content))

    except Exception as e:
        print(f"Error processing {module_path}: {str(e)}")


def main():
    """Main function to generate all API documentation."""
    # Setup paths
    docs_dir = Path("docs/api")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Generate docs for main API modules
    api_modules = [
        "src/bible_manager/manager.py",
        "src/contextual/analysis.py",
        "src/hermeneutics/interpreter.py",
        "src/lexicon/lexicon.py",
        "src/model/bible_model.py",
        "src/theology/validator.py",
        "app.py",
    ]

    for module in api_modules:
        generate_module_docs(module, docs_dir)

    # Generate index page
    index_content = [
        "# API Documentation\n",
        "This section contains the automatically generated API documentation for the Bible AI project.\n",
        "## Modules\n",
    ]

    for module in api_modules:
        module_name = module.split("/")[-1].replace(".py", "")
        index_content.append(f"- [{module_name.title()}]({module_name}.md)")

    index_path = docs_dir / "index.md"
    index_path.write_text("\n".join(index_content))


if __name__ == "__main__":
    main()
