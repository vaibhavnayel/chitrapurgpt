/**
 * Diacritical Translator
 * JavaScript module for the translator functionality
 */

// Main module for translator
const TranslatorModule = (function() {
    // Private variables
    let translationTimeout;
    let lastInputText = '';
    
    // Show notification
    function showNotification(message, type = 'success') {
        const notificationContainer = document.getElementById('notifications');
        if (!notificationContainer) {
            // Create notification container if it doesn't exist
            const container = document.createElement('div');
            container.id = 'notifications';
            container.className = 'notification-container';
            document.body.appendChild(container);
        }
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        const icon = type === 'success' ? 'check-circle' : 
                    type === 'error' ? 'exclamation-circle' :
                    type === 'warning' ? 'exclamation-triangle' : 'info-circle';
        
        notification.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        `;
        
        document.getElementById('notifications').appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }
    
    // Translate text function
    function translateText() {
        const inputText = document.getElementById('inputText').value;
        if (!inputText.trim()) {
            showNotification('Please enter some text to translate', 'warning');
            document.getElementById('inputText').focus();
            return;
        }
        
        // Show loading indicator
        const translateBtn = document.getElementById('translateBtn');
        const originalBtnText = translateBtn.innerHTML;
        translateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Translating...';
        translateBtn.disabled = true;
        
        // Add loading class to output
        const outputContainer = document.getElementById('outputText').closest('.text-panel');
        outputContainer.classList.add('loading');
        
        fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: inputText })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showNotification(data.error, 'error');
            } else {
                const outputText = document.getElementById('outputText');
                
                // Animate the text change
                outputText.style.opacity = '0';
                setTimeout(() => {
                    outputText.value = data.result;
                    outputText.style.opacity = '1';
                    
                    // Save the last input text
                    lastInputText = inputText;
                    
                    // Show success message
                    showNotification('Translation complete!');
                }, 300);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('An error occurred while translating', 'error');
        })
        .finally(() => {
            // Reset button state
            translateBtn.innerHTML = originalBtnText;
            translateBtn.disabled = false;
            
            // Remove loading class
            outputContainer.classList.remove('loading');
        });
    }

    // Copy to clipboard function
    function copyToClipboard() {
        const outputText = document.getElementById('outputText');
        if (!outputText.value.trim()) {
            showNotification('No text to copy', 'warning');
            return;
        }
        
        // Show animation on button
        const copyBtn = document.getElementById('copyBtn');
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.classList.add('btn-success');
        
        outputText.select();
        try {
            document.execCommand('copy');
            showNotification('Text copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy text:', err);
            // Fallback to modern clipboard API
            navigator.clipboard.writeText(outputText.value)
                .then(() => showNotification('Text copied to clipboard!'))
                .catch(err => {
                    console.error('Failed to copy text:', err);
                    showNotification('Failed to copy text', 'error');
                });
        }
        
        // Reset button after 2 seconds
        setTimeout(() => {
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            copyBtn.classList.remove('btn-success');
        }, 2000);
    }
    
    // Auto-translate after a delay
    function setupAutoTranslate() {
        const inputText = document.getElementById('inputText');
        
        inputText.addEventListener('input', function() {
            // Clear previous timeout
            if (translationTimeout) {
                clearTimeout(translationTimeout);
            }
            
            // Set new timeout for 1 second after typing stops
            translationTimeout = setTimeout(() => {
                if (inputText.value.trim() && inputText.value !== lastInputText) {
                    translateText();
                }
            }, 1000);
        });
    }
    
    // Setup keyboard shortcuts
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Ctrl+Enter to translate
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                translateText();
            }
            
            // Ctrl+Shift+C to copy
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'c') {
                e.preventDefault();
                copyToClipboard();
            }
        });
    }
    
    // Setup text area auto-resize
    function setupTextareaAutoResize() {
        const textareas = document.querySelectorAll('textarea');
        
        textareas.forEach(textarea => {
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
            
            // Initial resize
            textarea.style.height = 'auto';
            textarea.style.height = (textarea.scrollHeight) + 'px';
        });
    }

    // Public methods
    return {
        init: function() {
            // Set up event listeners
            document.getElementById('translateBtn').addEventListener('click', translateText);
            document.getElementById('copyBtn').addEventListener('click', copyToClipboard);
            
            // Setup additional features
            setupAutoTranslate();
            setupKeyboardShortcuts();
            setupTextareaAutoResize();
            
            // Show welcome message
            setTimeout(() => {
                showNotification('Welcome to Diacritical Translator!', 'info');
            }, 500);
        },
        translate: translateText,
        copyToClipboard: copyToClipboard
    };
})();

// Initialize when document is ready
$(document).ready(function() {
    TranslatorModule.init();
    
    // Add tooltips
    $('[title]').tooltip();
}); 