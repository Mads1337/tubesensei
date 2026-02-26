"""
Template helper functions for admin interface
"""

import re

import markdown as md
from markupsafe import Markup


def url_for(name: str, **kwargs):
    """URL generation helper for templates"""
    if name == 'static':
        return f"/static{kwargs.get('path', '')}"
    return "#"

def get_flashed_messages(with_categories=False):
    """Placeholder for flash messages"""
    return []

def _ensure_blank_line_before_lists(text):
    """Insert a blank line before list items that follow a non-list, non-blank line.

    Standard markdown requires a blank line before a list when it follows a
    paragraph.  LLM output often omits this, e.g. ``**Label:**\\n- item``.
    """
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        is_list = bool(re.match(r'^[ \t]*(?:[-*+]|\d+\.)[ \t]', line))
        if is_list and i > 0:
            prev = lines[i - 1]
            prev_is_list = bool(re.match(r'^[ \t]*(?:[-*+]|\d+\.)[ \t]', prev))
            prev_is_blank = prev.strip() == ''
            if not prev_is_list and not prev_is_blank:
                result.append('')
        result.append(line)
    return '\n'.join(result)

def render_markdown(text):
    """Convert markdown text to HTML."""
    if not text:
        return ""
    text = _ensure_blank_line_before_lists(text)
    return Markup(md.markdown(text, extensions=["fenced_code", "tables"]))

def register_markdown_filter(templates):
    """Register the render_markdown filter on a Jinja2Templates instance."""
    templates.env.filters["render_markdown"] = render_markdown

def get_template_context(request, **kwargs):
    """Get base template context with helper functions"""
    context = {
        "request": request,
        "url_for": url_for,
        "get_flashed_messages": get_flashed_messages,
    }
    context.update(kwargs)
    return context