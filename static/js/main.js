// TubeSensei Main JavaScript

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('TubeSensei Admin initialized');
    
    // Initialize tooltips
    initTooltips();
    
    // Initialize modals
    initModals();
    
    // Initialize form validations
    initFormValidation();
    
    // Initialize data tables
    initDataTables();
    
    // Initialize charts if present
    initCharts();
});

// Tooltip initialization
function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(tooltip => {
        // Tooltips are handled via CSS, this is for any JS enhancements
    });
}

// Modal functions
function initModals() {
    // Close modal when clicking backdrop
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal-backdrop')) {
            closeModal();
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    });
}

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal() {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.classList.add('hidden');
    });
    document.body.style.overflow = '';
}

// Form validation
function initFormValidation() {
    const forms = document.querySelectorAll('form[data-validate]');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(form)) {
                e.preventDefault();
            }
        });
    });
}

function validateForm(form) {
    let isValid = true;
    const requiredFields = form.querySelectorAll('[required]');
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showFieldError(field, 'This field is required');
            isValid = false;
        } else {
            clearFieldError(field);
        }
    });
    
    // Email validation
    const emailFields = form.querySelectorAll('input[type="email"]');
    emailFields.forEach(field => {
        if (field.value && !isValidEmail(field.value)) {
            showFieldError(field, 'Please enter a valid email address');
            isValid = false;
        }
    });
    
    return isValid;
}

function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function showFieldError(field, message) {
    field.classList.add('border-red-500');
    
    let errorElement = field.nextElementSibling;
    if (!errorElement || !errorElement.classList.contains('field-error')) {
        errorElement = document.createElement('span');
        errorElement.classList.add('field-error', 'text-red-500', 'text-sm', 'mt-1');
        field.parentNode.insertBefore(errorElement, field.nextSibling);
    }
    errorElement.textContent = message;
}

function clearFieldError(field) {
    field.classList.remove('border-red-500');
    
    const errorElement = field.nextElementSibling;
    if (errorElement && errorElement.classList.contains('field-error')) {
        errorElement.remove();
    }
}

// Data tables
function initDataTables() {
    const tables = document.querySelectorAll('[data-table]');
    tables.forEach(table => {
        // Add sorting functionality
        const headers = table.querySelectorAll('th[data-sortable]');
        headers.forEach(header => {
            header.addEventListener('click', function() {
                sortTable(table, header);
            });
        });
    });
}

function sortTable(table, header) {
    const columnIndex = Array.from(header.parentNode.children).indexOf(header);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const sortOrder = header.dataset.sortOrder === 'asc' ? 'desc' : 'asc';
    
    rows.sort((a, b) => {
        const aValue = a.children[columnIndex].textContent.trim();
        const bValue = b.children[columnIndex].textContent.trim();
        
        // Try to parse as number
        const aNum = parseFloat(aValue);
        const bNum = parseFloat(bValue);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return sortOrder === 'asc' ? aNum - bNum : bNum - aNum;
        }
        
        // Sort as string
        if (sortOrder === 'asc') {
            return aValue.localeCompare(bValue);
        } else {
            return bValue.localeCompare(aValue);
        }
    });
    
    // Update table
    tbody.innerHTML = '';
    rows.forEach(row => tbody.appendChild(row));
    
    // Update sort indicators
    table.querySelectorAll('th[data-sortable]').forEach(th => {
        th.dataset.sortOrder = '';
        th.classList.remove('sorted-asc', 'sorted-desc');
    });
    header.dataset.sortOrder = sortOrder;
    header.classList.add(`sorted-${sortOrder}`);
}

// Charts
function initCharts() {
    // Initialize any Chart.js charts
    const chartContainers = document.querySelectorAll('[data-chart]');
    chartContainers.forEach(container => {
        const chartType = container.dataset.chartType || 'line';
        const chartData = JSON.parse(container.dataset.chartData || '{}');
        const canvas = container.querySelector('canvas');
        
        if (canvas && window.Chart) {
            new Chart(canvas.getContext('2d'), {
                type: chartType,
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: container.dataset.showLegend !== 'false'
                        }
                    }
                }
            });
        }
    });
}

// Utility functions
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

function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Copy to clipboard
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Copied to clipboard', 'success');
        });
    } else {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showToast('Copied to clipboard', 'success');
    }
}

// Format numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Format dates
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

// Format time ago
function timeAgo(dateString) {
    const date = new Date(dateString);
    const seconds = Math.floor((new Date() - date) / 1000);
    
    const intervals = {
        year: 31536000,
        month: 2592000,
        week: 604800,
        day: 86400,
        hour: 3600,
        minute: 60
    };
    
    for (const [unit, secondsInUnit] of Object.entries(intervals)) {
        const interval = Math.floor(seconds / secondsInUnit);
        if (interval >= 1) {
            return `${interval} ${unit}${interval > 1 ? 's' : ''} ago`;
        }
    }
    
    return 'just now';
}

// Confirmation dialog
function confirmAction(message, callback) {
    if (confirm(message)) {
        callback();
    }
}

// File upload handler
function handleFileUpload(input, callback) {
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            callback(e.target.result, file);
        };
        reader.readAsDataURL(file);
    }
}

// Drag and drop
function initDragAndDrop(element, callback) {
    element.addEventListener('dragover', (e) => {
        e.preventDefault();
        element.classList.add('dragover');
    });
    
    element.addEventListener('dragleave', () => {
        element.classList.remove('dragover');
    });
    
    element.addEventListener('drop', (e) => {
        e.preventDefault();
        element.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            callback(files);
        }
    });
}

// Search functionality
function initSearch(searchInput, searchableElements) {
    searchInput.addEventListener('input', debounce(function() {
        const searchTerm = this.value.toLowerCase();
        
        searchableElements.forEach(element => {
            const text = element.textContent.toLowerCase();
            if (text.includes(searchTerm)) {
                element.style.display = '';
            } else {
                element.style.display = 'none';
            }
        });
    }, 300));
}

// Export functions for global use
window.TubeSensei = {
    openModal,
    closeModal,
    showToast,
    copyToClipboard,
    formatNumber,
    formatDate,
    timeAgo,
    confirmAction,
    handleFileUpload,
    initDragAndDrop,
    initSearch
};