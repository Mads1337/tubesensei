"""
Template helper functions for admin interface
"""

def url_for(name: str, **kwargs):
    """URL generation helper for templates"""
    if name == 'static':
        return f"/static{kwargs.get('path', '')}"
    return "#"

def get_flashed_messages(with_categories=False):
    """Placeholder for flash messages"""
    return []

def get_template_context(request, **kwargs):
    """Get base template context with helper functions"""
    context = {
        "request": request,
        "url_for": url_for,
        "get_flashed_messages": get_flashed_messages,
    }
    context.update(kwargs)
    return context