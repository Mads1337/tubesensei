// TubeSensei Admin Panel JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('TubeSensei Admin Panel loaded');
    
    // Initialize tooltips if any
    initializeTooltips();
    
    // Setup keyboard shortcuts
    setupKeyboardShortcuts();
    
    // Initialize auto-refresh functionality
    initializeAutoRefresh();
});

/**
 * Initialize tooltips for elements with data-tooltip attribute
 */
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

/**
 * Show tooltip
 */
function showTooltip(event) {
    const element = event.target;
    const tooltipText = element.getAttribute('data-tooltip');
    
    if (!tooltipText) return;
    
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip absolute z-50 px-2 py-1 text-xs text-white bg-gray-900 rounded shadow-lg';
    tooltip.textContent = tooltipText;
    tooltip.id = 'tooltip';
    
    document.body.appendChild(tooltip);
    
    const rect = element.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
}

/**
 * Hide tooltip
 */
function hideTooltip() {
    const tooltip = document.getElementById('tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Ctrl/Cmd + K for search
        if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
            event.preventDefault();
            const searchInput = document.querySelector('input[placeholder*="Search"]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to close modals
        if (event.key === 'Escape') {
            closeModals();
        }
        
        // Ctrl/Cmd + N for new channel (on channels page)
        if ((event.ctrlKey || event.metaKey) && event.key === 'n' && window.location.pathname.includes('/channels')) {
            event.preventDefault();
            const addButton = document.querySelector('[hx-get*="/add"]');
            if (addButton) {
                addButton.click();
            }
        }
    });
}

/**
 * Close all open modals
 */
function closeModals() {
    // Close Alpine.js dropdowns
    document.querySelectorAll('[x-data]').forEach(element => {
        if (element._x_dataStack && element._x_dataStack[0].open) {
            element._x_dataStack[0].open = false;
        }
    });
    
    // Close custom modals
    const modals = document.querySelectorAll('.modal-backdrop:not(.hidden)');
    modals.forEach(modal => {
        modal.classList.add('hidden');
    });
    
    // Clear modal container
    const modalContainer = document.getElementById('modal-container');
    if (modalContainer) {
        modalContainer.innerHTML = '';
    }
}

/**
 * Initialize auto-refresh for dynamic content
 */
function initializeAutoRefresh() {
    // Auto-refresh channel cards every 5 minutes
    if (window.location.pathname.includes('/channels')) {
        setInterval(() => {
            if (!document.hidden) {
                const channelsGrid = document.getElementById('channels-grid');
                if (channelsGrid && !document.querySelector('.htmx-request')) {
                    // Trigger a gentle refresh
                    htmx.trigger(channelsGrid, 'refresh');
                }
            }
        }, 300000); // 5 minutes
    }
}

/**
 * Show toast notification
 * @param {string} message - The message to show
 * @param {string} type - success, error, warning, info
 * @param {number} duration - Duration in milliseconds (default: 5000)
 */
function showToast(message, type = 'info', duration = 5000) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;
    
    const toast = document.createElement('div');
    toast.className = `p-4 rounded-lg shadow-lg flex items-center space-x-3 mb-2 toast-${type} animate-fadeIn`;
    
    // Set colors based on type
    const colors = {
        success: 'bg-green-500 text-white',
        error: 'bg-red-500 text-white',
        warning: 'bg-yellow-500 text-white',
        info: 'bg-blue-500 text-white'
    };
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    toast.className += ' ' + colors[type];
    
    toast.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <span class="flex-1">${message}</span>
        <button onclick="this.parentElement.remove()" class="ml-4 opacity-75 hover:opacity-100">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }
    }, duration);
}

/**
 * Format numbers with appropriate suffixes (K, M, B)
 * @param {number} num - The number to format
 * @returns {string} - Formatted number
 */
function formatNumber(num) {
    if (num >= 1000000000) {
        return (num / 1000000000).toFixed(1) + 'B';
    }
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

/**
 * Debounce function to limit the rate of function execution
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} - Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 */
function copyToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Copied to clipboard!', 'success', 2000);
        }).catch(() => {
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
}

/**
 * Fallback copy to clipboard for older browsers
 * @param {string} text - Text to copy
 */
function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showToast('Copied to clipboard!', 'success', 2000);
    } catch (err) {
        showToast('Failed to copy to clipboard', 'error', 3000);
    }
    
    document.body.removeChild(textArea);
}

/**
 * Confirm action with a custom dialog
 * @param {string} message - Confirmation message
 * @param {Function} onConfirm - Function to execute on confirmation
 * @param {string} confirmText - Text for confirm button (default: 'Confirm')
 * @param {string} cancelText - Text for cancel button (default: 'Cancel')
 */
function confirmAction(message, onConfirm, confirmText = 'Confirm', cancelText = 'Cancel') {
    const modal = document.createElement('div');
    modal.className = 'modal-backdrop';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="text-center">
                <div class="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-yellow-100 mb-4">
                    <i class="fas fa-exclamation-triangle text-yellow-600 text-xl"></i>
                </div>
                <h3 class="text-lg font-medium text-gray-900 mb-2">Confirm Action</h3>
                <p class="text-sm text-gray-500 mb-6">${message}</p>
                <div class="flex justify-center space-x-4">
                    <button class="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400" onclick="this.closest('.modal-backdrop').remove()">
                        ${cancelText}
                    </button>
                    <button class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700" onclick="confirmAndClose(this, arguments[0])">
                        ${confirmText}
                    </button>
                </div>
            </div>
        </div>
    `;
    
    // Store the callback function on the confirm button
    modal.querySelector('.bg-red-600').onConfirm = onConfirm;
    
    document.body.appendChild(modal);
}

/**
 * Execute confirmation callback and close modal
 * @param {HTMLElement} button - The confirm button
 * @param {Function} callback - The callback to execute
 */
function confirmAndClose(button) {
    const modal = button.closest('.modal-backdrop');
    if (button.onConfirm) {
        button.onConfirm();
    }
    modal.remove();
}

/**
 * Initialize charts (placeholder for future chart implementation)
 */
function initializeCharts() {
    // This will be implemented when we add actual charts
    console.log('Chart initialization placeholder');
}

/**
 * Validate form inputs
 * @param {HTMLFormElement} form - The form to validate
 * @returns {boolean} - True if valid, false otherwise
 */
function validateForm(form) {
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('border-red-500');
            isValid = false;
        } else {
            input.classList.remove('border-red-500');
        }
    });
    
    return isValid;
}

// HTMX Event Listeners
document.body.addEventListener('htmx:beforeRequest', function(event) {
    // Show loading indicator
    const indicator = document.getElementById('loading-indicator');
    if (indicator) {
        indicator.style.display = 'block';
    }
});

document.body.addEventListener('htmx:afterRequest', function(event) {
    // Hide loading indicator
    const indicator = document.getElementById('loading-indicator');
    if (indicator) {
        indicator.style.display = 'none';
    }
    
    // Show toast messages from headers
    if (event.detail.xhr.getResponseHeader('X-Toast')) {
        try {
            const toast = JSON.parse(event.detail.xhr.getResponseHeader('X-Toast'));
            showToast(toast.message, toast.type || 'info');
        } catch (e) {
            console.warn('Failed to parse toast message from header');
        }
    }
});

document.body.addEventListener('htmx:responseError', function(event) {
    showToast('Request failed. Please try again.', 'error');
});

document.body.addEventListener('htmx:sendError', function(event) {
    showToast('Network error. Please check your connection.', 'error');
});

// Export functions for global use
window.TubeSensei = {
    showToast,
    formatNumber,
    copyToClipboard,
    confirmAction,
    debounce
};