/**
 * Diacritical Mappings Management
 * JavaScript module for managing diacritical mappings
 */

// Main module for mappings management
const MappingsManager = (function() {
    // Private variables
    let mappingsTable;
    let deleteId = null;
    let selectedRows = [];
    
    // Initialize DataTable
    function initializeTable() {
        mappingsTable = $('#mappingsTable').DataTable({
            ajax: {
                url: '/api/mappings',
                dataSrc: '',
                beforeSend: function() {
                    // Show loading indicator
                    $('#mappingsTable').addClass('loading');
                    $('body').append('<div class="table-loading-overlay"><div class="spinner-border text-primary" role="status"><span class="sr-only">Loading...</span></div></div>');
                },
                complete: function() {
                    // Hide loading indicator
                    $('#mappingsTable').removeClass('loading');
                    $('.table-loading-overlay').remove();
                }
            },
            columns: [
                {
                    data: null,
                    defaultContent: '',
                    orderable: false,
                    className: 'select-checkbox'
                },
                { data: 'id' },
                { data: 'plain_text' },
                { data: 'diacritic_text' },
                { 
                    data: 'created_at',
                    render: function(data) {
                        return new Date(data).toLocaleString();
                    }
                },
                { 
                    data: 'updated_at',
                    render: function(data) {
                        return new Date(data).toLocaleString();
                    }
                },
                {
                    data: null,
                    className: 'action-buttons',
                    orderable: false,
                    render: function(data) {
                        return `
                            <button class="btn btn-sm btn-primary edit-btn" data-id="${data.id}" title="Edit Mapping">
                                <i class="fas fa-edit"></i>
                            </button>
                            <button class="btn btn-sm btn-danger delete-btn" data-id="${data.id}" title="Delete Mapping">
                                <i class="fas fa-trash-alt"></i>
                            </button>
                        `;
                    }
                }
            ],
            order: [[1, 'asc']],
            select: {
                style: 'multi',
                selector: 'td:first-child'
            },
            pageLength: 10,
            lengthMenu: [10, 25, 50, 100],
            responsive: true,
            language: {
                emptyTable: "No mappings found. Add your first mapping above!",
                zeroRecords: "No matching mappings found",
                info: "Showing _START_ to _END_ of _TOTAL_ mappings",
                infoEmpty: "Showing 0 to 0 of 0 mappings",
                infoFiltered: "(filtered from _MAX_ total mappings)",
                search: "<i class='fas fa-search'></i> Search:",
                paginate: {
                    first: "<i class='fas fa-angle-double-left'></i>",
                    previous: "<i class='fas fa-angle-left'></i>",
                    next: "<i class='fas fa-angle-right'></i>",
                    last: "<i class='fas fa-angle-double-right'></i>"
                }
            },
            drawCallback: function() {
                // Add tooltip to buttons
                $('[title]').tooltip();
            }
        });
        
        // Handle row selection
        mappingsTable.on('select', function() {
            selectedRows = mappingsTable.rows({ selected: true }).data().toArray();
            updateSelectedCount();
        });
        
        mappingsTable.on('deselect', function() {
            selectedRows = mappingsTable.rows({ selected: true }).data().toArray();
            updateSelectedCount();
        });
        
        // Handle page length change
        $('#pageLengthSelect').on('change', function() {
            const newLength = parseInt($(this).val(), 10);
            mappingsTable.page.len(newLength).draw();
        });
    }
    
    // Update selected count and button state
    function updateSelectedCount() {
        const count = selectedRows.length;
        $('#selectedCount').text(`(${count} item${count !== 1 ? 's' : ''} selected)`);
        $('#batchDeleteBtn').prop('disabled', count === 0);
        
        // Animate count change
        $('#selectedCount').addClass('highlight');
        setTimeout(() => {
            $('#selectedCount').removeClass('highlight');
        }, 300);
    }
    
    // Show alert message
    function showAlert(message, type = 'success') {
        const icon = type === 'success' ? 'check-circle' : 
                    type === 'danger' ? 'exclamation-circle' :
                    type === 'warning' ? 'exclamation-triangle' : 'info-circle';
                    
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                <i class="fas fa-${icon} mr-2"></i> ${message}
                <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
        `;
        
        // Remove existing alerts
        $('#alerts').html('');
        
        // Add new alert with animation
        const $alert = $(alertHtml).appendTo('#alerts');
        $alert.hide().slideDown(300);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            $alert.slideUp(300, function() {
                $(this).remove();
            });
        }, 5000);
    }
    
    // Add new mapping
    function setupAddMappingForm() {
        $('#mappingForm').on('submit', function(e) {
            e.preventDefault();
            
            // Convert inputs to lowercase
            const plainText = $('#plainText').val().trim().toLowerCase();
            const diacriticText = $('#diacriticText').val().trim().toLowerCase();
            
            // Update the input fields with lowercase values
            $('#plainText').val(plainText);
            $('#diacriticText').val(diacriticText);
            
            // Disable form during submission
            const $form = $(this);
            const $submitBtn = $form.find('button[type="submit"]');
            const originalBtnText = $submitBtn.html();
            
            $form.find('input').prop('disabled', true);
            $submitBtn.html('<i class="fas fa-spinner fa-spin"></i> Adding...');
            $submitBtn.prop('disabled', true);
            
            $.ajax({
                url: '/api/mappings',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    plain_text: plainText,
                    diacritic_text: diacriticText
                }),
                success: function(response) {
                    showAlert('Mapping added successfully!');
                    $form[0].reset();
                    
                    // Highlight the new row
                    mappingsTable.ajax.reload(function() {
                        highlightNewMapping(response.id);
                    });
                },
                error: function(xhr) {
                    const error = xhr.responseJSON ? xhr.responseJSON.error : 'An error occurred';
                    showAlert(error, 'danger');
                },
                complete: function() {
                    // Re-enable form
                    $form.find('input').prop('disabled', false);
                    $submitBtn.html(originalBtnText);
                    $submitBtn.prop('disabled', false);
                    $('#plainText').focus();
                }
            });
        });
        
        // Add input event listeners to convert text to lowercase as user types
        $('#plainText, #diacriticText').on('input', function() {
            const input = $(this);
            const cursorPosition = this.selectionStart;
            const value = input.val();
            const lowercaseValue = value.toLowerCase();
            
            // Only update if there's a difference to avoid cursor jumping
            if (value !== lowercaseValue) {
                input.val(lowercaseValue);
                // Restore cursor position
                this.setSelectionRange(cursorPosition, cursorPosition);
            }
        });
        
        // Also convert on paste events
        $('#plainText, #diacriticText').on('paste', function() {
            const input = $(this);
            // Use setTimeout to get the value after the paste event completes
            setTimeout(function() {
                const value = input.val();
                input.val(value.toLowerCase());
            }, 0);
        });
    }
    
    // Highlight a newly added or updated mapping
    function highlightNewMapping(id) {
        // Find the row with the matching ID
        const $rows = $('#mappingsTable tbody tr');
        let $targetRow = null;
        
        $rows.each(function() {
            const rowData = mappingsTable.row(this).data();
            if (rowData && rowData.id === id) {
                $targetRow = $(this);
                return false; // Break the loop
            }
        });
        
        if ($targetRow) {
            // Scroll to the row if needed
            const $tableContainer = $('#mappingsTable').closest('.table-responsive');
            const rowTop = $targetRow.position().top;
            const containerHeight = $tableContainer.height();
            const scrollTop = $tableContainer.scrollTop();
            
            if (rowTop < 0 || rowTop > containerHeight) {
                $tableContainer.animate({
                    scrollTop: scrollTop + rowTop - 50
                }, 300);
            }
            
            // Highlight the row
            $targetRow.addClass('highlight-row');
            setTimeout(() => {
                $targetRow.removeClass('highlight-row');
            }, 2000);
        }
    }
    
    // Edit mapping functionality
    function setupEditMapping() {
        // Open edit modal
        $('#mappingsTable').on('click', '.edit-btn', function() {
            const id = $(this).data('id');
            const row = mappingsTable.row($(this).closest('tr')).data();
            
            $('#editId').val(id);
            $('#editPlainText').val(row.plain_text.toLowerCase());
            $('#editDiacriticText').val(row.diacritic_text.toLowerCase());
            
            $('#editModal').modal('show');
            setTimeout(() => {
                $('#editPlainText').focus();
            }, 500);
        });
        
        // Save edited mapping
        $('#saveEdit').on('click', function() {
            const id = $('#editId').val();
            
            // Convert inputs to lowercase
            const plainText = $('#editPlainText').val().trim().toLowerCase();
            const diacriticText = $('#editDiacriticText').val().trim().toLowerCase();
            
            // Update the input fields with lowercase values
            $('#editPlainText').val(plainText);
            $('#editDiacriticText').val(diacriticText);
            
            // Disable button during submission
            const $btn = $(this);
            const originalBtnText = $btn.html();
            $btn.html('<i class="fas fa-spinner fa-spin"></i> Saving...');
            $btn.prop('disabled', true);
            
            $.ajax({
                url: `/api/mappings/${id}`,
                method: 'PUT',
                contentType: 'application/json',
                data: JSON.stringify({
                    plain_text: plainText,
                    diacritic_text: diacriticText
                }),
                success: function(response) {
                    $('#editModal').modal('hide');
                    showAlert('Mapping updated successfully!');
                    
                    // Highlight the updated row
                    mappingsTable.ajax.reload(function() {
                        highlightNewMapping(id);
                    });
                },
                error: function(xhr) {
                    const error = xhr.responseJSON ? xhr.responseJSON.error : 'An error occurred';
                    showAlert(error, 'danger');
                },
                complete: function() {
                    // Re-enable button
                    $btn.html(originalBtnText);
                    $btn.prop('disabled', false);
                }
            });
        });
        
        // Handle enter key in edit modal
        $('#editForm').on('keypress', function(e) {
            if (e.which === 13) {
                e.preventDefault();
                $('#saveEdit').click();
            }
        });
        
        // Add input event listeners to convert text to lowercase as user types in edit modal
        $('#editPlainText, #editDiacriticText').on('input', function() {
            const input = $(this);
            const cursorPosition = this.selectionStart;
            const value = input.val();
            const lowercaseValue = value.toLowerCase();
            
            // Only update if there's a difference to avoid cursor jumping
            if (value !== lowercaseValue) {
                input.val(lowercaseValue);
                // Restore cursor position
                this.setSelectionRange(cursorPosition, cursorPosition);
            }
        });
        
        // Also convert on paste events in edit modal
        $('#editPlainText, #editDiacriticText').on('paste', function() {
            const input = $(this);
            // Use setTimeout to get the value after the paste event completes
            setTimeout(function() {
                const value = input.val();
                input.val(value.toLowerCase());
            }, 0);
        });
    }
    
    // Delete mapping functionality
    function setupDeleteMapping() {
        // Open delete confirmation modal
        $('#mappingsTable').on('click', '.delete-btn', function() {
            const id = $(this).data('id');
            const row = mappingsTable.row($(this).closest('tr')).data();
            
            deleteId = id;
            $('#deletePlainText').text(row.plain_text);
            $('#deleteDiacriticText').text(row.diacritic_text);
            
            $('#deleteModal').modal('show');
        });
        
        // Confirm delete mapping
        $('#confirmDelete').on('click', function() {
            if (!deleteId) return;
            
            // Disable button during submission
            const $btn = $(this);
            const originalBtnText = $btn.html();
            $btn.html('<i class="fas fa-spinner fa-spin"></i> Deleting...');
            $btn.prop('disabled', true);
            
            $.ajax({
                url: `/api/mappings/${deleteId}`,
                method: 'DELETE',
                success: function() {
                    $('#deleteModal').modal('hide');
                    showAlert('Mapping deleted successfully!');
                    mappingsTable.ajax.reload();
                },
                error: function(xhr) {
                    $('#deleteModal').modal('hide');
                    const error = xhr.responseJSON ? xhr.responseJSON.error : 'An error occurred';
                    showAlert(error, 'danger');
                },
                complete: function() {
                    // Re-enable button
                    $btn.html(originalBtnText);
                    $btn.prop('disabled', false);
                }
            });
        });
    }
    
    // Batch delete functionality
    function setupBatchDelete() {
        // Open batch delete confirmation modal
        $('#batchDeleteBtn').on('click', function() {
            const count = selectedRows.length;
            if (count === 0) return;
            
            $('#deleteCount').text(count);
            $('#batchDeleteModal').modal('show');
        });
        
        // Confirm batch delete
        $('#confirmBatchDelete').on('click', function() {
            if (selectedRows.length === 0) return;
            
            const ids = selectedRows.map(row => row.id);
            
            // Disable button during submission
            const $btn = $(this);
            const originalBtnText = $btn.html();
            $btn.html('<i class="fas fa-spinner fa-spin"></i> Deleting...');
            $btn.prop('disabled', true);
            
            $.ajax({
                url: '/api/mappings/batch-delete',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ ids: ids }),
                success: function(response) {
                    $('#batchDeleteModal').modal('hide');
                    showAlert(`Successfully deleted ${response.deleted_count} mappings!`);
                    mappingsTable.ajax.reload();
                    selectedRows = [];
                    updateSelectedCount();
                },
                error: function(xhr) {
                    $('#batchDeleteModal').modal('hide');
                    const error = xhr.responseJSON ? xhr.responseJSON.error : 'An error occurred';
                    showAlert(error, 'danger');
                },
                complete: function() {
                    // Re-enable button
                    $btn.html(originalBtnText);
                    $btn.prop('disabled', false);
                }
            });
        });
    }
    
    // File upload functionality
    function setupFileUpload() {
        $('#uploadForm').on('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('mappingsFile');
            if (!fileInput.files.length) {
                showAlert('Please select a file to upload', 'danger');
                return;
            }
            
            const file = fileInput.files[0];
            const importMode = $('input[name="importMode"]:checked').val();
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('mode', importMode);
            
            // Disable form during submission
            const $form = $(this);
            const $submitBtn = $form.find('button[type="submit"]');
            const originalBtnText = $submitBtn.html();
            
            $form.find('input').prop('disabled', true);
            $submitBtn.html('<i class="fas fa-spinner fa-spin"></i> Uploading...');
            $submitBtn.prop('disabled', true);
            
            // Show progress and status
            $('#uploadProgress').show();
            $('#uploadStatus').html('<p class="text-info"><i class="fas fa-info-circle"></i> Uploading file...</p>');
            
            $.ajax({
                url: '/api/mappings/upload',
                method: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function(response) {
                    $('#uploadProgress').hide();
                    $('#uploadStatus').html(`<p class="text-success"><i class="fas fa-check-circle"></i> File processed successfully!</p>`);
                    
                    let message = '';
                    if (importMode === 'update') {
                        message = `Added ${response.added} new mappings, ${response.updated} updated.`;
                    } else {
                        message = `Database reset with ${response.total} mappings.`;
                    }
                    
                    showAlert(message);
                    mappingsTable.ajax.reload();
                    $form[0].reset();
                },
                error: function(xhr) {
                    $('#uploadProgress').hide();
                    $('#uploadStatus').html(`<p class="text-danger"><i class="fas fa-exclamation-circle"></i> Upload failed. Please try again.</p>`);
                    const error = xhr.responseJSON ? xhr.responseJSON.error : 'An error occurred';
                    showAlert(error, 'danger');
                },
                complete: function() {
                    // Re-enable form
                    $form.find('input').prop('disabled', false);
                    $submitBtn.html(originalBtnText);
                    $submitBtn.prop('disabled', false);
                }
            });
        });
        
        // Clear status when selecting a new file
        $('#mappingsFile').on('change', function() {
            $('#uploadStatus').html('');
        });
    }
    
    // Add keyboard shortcuts
    function setupKeyboardShortcuts() {
        $(document).on('keydown', function(e) {
            // Alt+N to focus on new mapping form
            if (e.altKey && e.key === 'n') {
                e.preventDefault();
                $('#plainText').focus();
            }
            
            // Alt+D to focus on delete selected button (if enabled)
            if (e.altKey && e.key === 'd') {
                e.preventDefault();
                if (!$('#batchDeleteBtn').prop('disabled')) {
                    $('#batchDeleteBtn').focus();
                }
            }
            
            // Alt+S to focus on search box
            if (e.altKey && e.key === 's') {
                e.preventDefault();
                $('.dataTables_filter input').focus();
            }
        });
    }
    
    // Public methods
    return {
        init: function() {
            initializeTable();
            setupAddMappingForm();
            setupEditMapping();
            setupDeleteMapping();
            setupBatchDelete();
            setupFileUpload();
            setupKeyboardShortcuts();
            
            // Show welcome message
            setTimeout(() => {
                showAlert('Welcome to Diacritical Mappings Manager!', 'info');
            }, 500);
        },
        refreshTable: function() {
            if (mappingsTable) {
                mappingsTable.ajax.reload();
            }
        },
        showAlert: showAlert
    };
})();

// Initialize when document is ready
$(document).ready(function() {
    MappingsManager.init();
    
    // Add tooltips to static elements
    $('[data-toggle="tooltip"]').tooltip();
    
    // Add animation to the download button
    $('#downloadBtn').hover(
        function() {
            $(this).find('i').addClass('fa-bounce');
        },
        function() {
            $(this).find('i').removeClass('fa-bounce');
        }
    );
}); 