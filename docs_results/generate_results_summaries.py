from jinja2 import Template
from pathlib import Path

# Load template
with open('_templates/entry_summary.rst.j2', 'r') as f:
    template = Template(f.read())

base_path = Path('results')
directories = [d.name for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]


# Generate RST file for each directory
for dir_name in directories:
    # Render template
    content = template.render(dir_name=dir_name)
    
    # Write to file
    output_path = base_path / Path(dir_name) / 'index.rst'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Generated: {output_path}")
