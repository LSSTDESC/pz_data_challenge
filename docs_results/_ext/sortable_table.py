from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util.docutils import SphinxDirective
import pandas as pd

class SortableTable(SphinxDirective):
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    option_spec = {
        'file': directives.path,
        'header-rows': directives.nonnegative_int,
    }

    def run(self):
        # Generate unique ID
        table_id = f"sortable-{self.env.new_serialno('sortable')}"

        df = pd.read_csv(self.options['file'])
        
        # Generate HTML table
        table_html = df.to_html(
            classes='display sortable-table',
            table_id=table_id,
            index=False
        )
        
        # Create raw HTML node
        raw_node = nodes.raw('', table_html, format='html')
        
        return [raw_node]

def setup(app):
    app.add_directive('sortable-table', SortableTable)
    app.add_js_file('https://code.jquery.com/jquery-3.6.0.min.js')
    app.add_js_file('https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js')
    app.add_css_file('https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css')
    app.add_js_file('sortable_init.js')
    
    return {'version': '0.1'}
