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
                url: `${window.DIACRITICAL_ROOT || ''}/api/mappings`,
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
    
    // The house style (the same as cleaning.py on the server; in mapping_form.js): the plain text loses its
    // diacritical marks, and both lose the spaces at their start and end (spaces between words stay, so phrases keep
    // working); the result's broken {{ }} brackets are repaired. The Choices box is cleaned line by line.
    const Codes = window.MappingCodes;
    const cleanPlain = Codes.cleanPlain;
    const cleanResult = Codes.cleanResult;

    function cleanerFor(input) {
        if (/plain/i.test(input.id)) return cleanPlain;
        return input.tagName === 'TEXTAREA' ? Codes.cleanChoicesText : cleanResult;
    }

    // The Type radios and boxes of one form (Add: prefix 'add', Edit: 'edit'); the old code box is under Advanced.
    function TypeForm(prefix, codeSelector, plainSelector) {
        const $root = $('#' + prefix + 'Boxes');
        const $result = $('#' + prefix + 'Result');
        const $name = $('#' + prefix + 'AsName');
        const $choices = $('#' + prefix + 'Choices');
        const $capital = $('#' + prefix + 'ChoicesCapital');
        const $code = $(codeSelector);
        const $plain = $(plainSelector);
        const $advanced = $('#' + prefix + 'Advanced');
        const $toggle = $('#' + prefix + 'AdvancedToggle');
        const $note = $('#' + prefix + 'AdvancedNote');
        const $problems = $('#' + prefix + 'Problems');
        let advancedOpen = false;
        let triedSaving = false;  // after a Save with problems, they stay shown until fixed (no jumping on blur)

        function type() {
            return $root.find('input[name="' + prefix + 'Type"]:checked').val() || 'normal';
        }

        function setType(value) {
            $('#' + prefix + 'Type-' + value).prop('checked', true);
        }

        // The optional "spelt differently when typed with a capital" box: for Normal, and Choices not always capital.
        function nameShown() {
            const t = type();
            return t === 'normal' || (t === 'choices' && !$capital.is(':checked'));
        }

        function boxes() {
            return {
                type: type(),
                result: $result.val(),
                name: nameShown() ? $name.val() : '',
                choices: Codes.splitChoiceLines($choices.val()),
                capital: $capital.is(':checked'),
                advanced: advancedOpen ? $code.val() : ''
            };
        }

        function usingAdvanced() {
            return advancedOpen && !!$code.val().trim();
        }

        function setAdvanced(open, note) {
            advancedOpen = open;
            $advanced.toggle(open);
            $toggle.attr('aria-expanded', open ? 'true' : 'false');
            $toggle.find('i').toggleClass('fa-caret-right', !open).toggleClass('fa-caret-down', open);
            $note.text(note || '').toggle(!!note);
        }

        function update() {
            const t = type();
            $root.find('.box-result').toggle(t !== 'choices');
            $root.find('.box-name').toggle(nameShown());
            $root.find('.box-choices').toggle(t === 'choices');
            const b = boxes();
            const code = Codes.boxesToCode(b);
            const adv = usingAdvanced();
            $root.toggleClass('advanced-wins', adv);
            const problems = adv ? [Codes.codeWarning(code)].filter(Boolean) : Codes.boxProblems(b, triedSaving);
            const typedWrong = !adv && Codes.boxProblems(b, false).length > 0;
            $('#' + prefix + 'Give').text((!typedWrong && Codes.describeCode(code, $plain.val())) || '\u2013');
            $('#' + prefix + 'SavedAs').text(code || '\u2013');
            $problems.text(problems.join('\n'));
        }

        // The problems that stop saving (shown under the boxes); none means the code can be saved.
        function check() {
            const problems = usingAdvanced() ? [] : Codes.boxProblems(boxes(), true);
            triedSaving = problems.length > 0;
            if (problems.length) {
                $problems.text(problems.join('\n'));
                const t = type();
                const $first = t === 'choices' ? (Codes.splitChoiceLines($choices.val()).length < 2 ? $choices : $name)
                    : (!$result.val().trim() || /[{},]/.test($result.val())) ? $result : $name;
                $first.focus();
            }
            return problems;
        }

        // Clean the boxes in place (as on leaving them) and give the code to save.
        function code() {
            [$result, $name, $code].forEach(($el) => $el.val(cleanResult($el.val())));
            $choices.val(Codes.cleanChoicesText($choices.val()));
            update();
            return Codes.boxesToCode(boxes());
        }

        function clear() {
            setType('normal');
            $result.val('');
            $name.val('');
            $choices.val('');
            $capital.prop('checked', false);
            $code.val('');
        }

        // Show a stored code in the boxes; a code they can't show exactly opens Advanced with it.
        function load(stored) {
            clear();
            triedSaving = false;
            const b = Codes.codeToBoxes(stored || '');
            if (b.type === 'advanced') {
                $code.val(b.advanced);
                setAdvanced(true, 'This entry can\'t be shown with the boxes above (' + b.why +
                    '), so its code is here. Change it here, or clear it and use the boxes.');
            } else {
                setType(b.type);
                $result.val(b.result);
                $name.val(b.name);
                $choices.val(b.choices.join('\n'));
                $capital.prop('checked', b.capital);
                setAdvanced(false);
            }
            update();
        }

        function reset() {
            clear();
            triedSaving = false;
            setAdvanced(false);
            update();
        }

        $toggle.on('click', function(e) {
            e.preventDefault();
            setAdvanced(!advancedOpen);
            update();
            if (advancedOpen) $code.focus();
        });
        $root.on('change', 'input[type=radio]', function() { triedSaving = false; });
        $root.on('input change', 'input, textarea', update);
        $plain.on('input change', update);
        update();

        return { load: load, reset: reset, check: check, code: code, update: update };
    }

    let addForm, editForm;

    // Add new mapping
    function setupAddMappingForm() {
        $('#mappingForm').on('submit', function(e) {
            e.preventDefault();
            
            // The house style: plain text without marks, the code from the boxes (or the Advanced box)
            const plainText = cleanPlain($('#plainText').val());
            $('#plainText').val(plainText);
            if (addForm.check().length) return;
            const diacriticText = addForm.code();
            
            // Disable form during submission
            const $form = $(this);
            const $submitBtn = $form.find('button[type="submit"]');
            const originalBtnText = $submitBtn.html();
            
            $form.find('input, textarea').prop('disabled', true);
            $submitBtn.html('<i class="fas fa-spinner fa-spin"></i> Adding...');
            $submitBtn.prop('disabled', true);
            
            $.ajax({
                url: `${window.DIACRITICAL_ROOT || ''}/api/mappings`,
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    plain_text: plainText,
                    diacritic_text: diacriticText
                }),
                success: function(response) {
                    showAlert('Mapping added successfully!');
                    $form[0].reset();
                    addForm.reset();
                    
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
                    $form.find('input, textarea').prop('disabled', false);
                    $submitBtn.html(originalBtnText);
                    $submitBtn.prop('disabled', false);
                    $('#plainText').focus();
                }
            });
        });
        
        // Add input event listeners to convert text to lowercase as user types
        $('#plainText, #diacriticText, #addResult, #addAsName, #addChoices').on('input', function() {
            const input = $(this);
            const cursorPosition = this.selectionStart;
            const value = input.val();
            let lowercaseValue = value.toLowerCase();
            if (/plain/i.test(this.id)) {
                lowercaseValue = lowercaseValue.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
            }
            
            // Only update if there's a difference to avoid cursor jumping
            if (value !== lowercaseValue) {
                input.val(lowercaseValue);
                // Restore cursor position
                this.setSelectionRange(cursorPosition, cursorPosition);
            }
        });
        
        // Also convert on paste events
        $('#plainText, #diacriticText, #addResult, #addAsName, #addChoices').on('paste', function() {
            const input = $(this);
            // Use setTimeout to get the value after the paste event completes
            setTimeout(function() {
                input.val(cleanerFor(input[0])(input.val()));
                addForm.update();
            }, 0);
        }).on('blur', function() {
            $(this).val(cleanerFor(this)($(this).val()));
            addForm.update();
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
            editForm.load(row.diacritic_text.toLowerCase());
            
            $('#editModal').modal('show');
            setTimeout(() => {
                $('#editPlainText').focus();
            }, 500);
        });
        
        // Save edited mapping
        $('#saveEdit').on('click', function() {
            const id = $('#editId').val();
            
            // The house style: plain text without marks, the code from the boxes (or the Advanced box)
            const plainText = cleanPlain($('#editPlainText').val());
            $('#editPlainText').val(plainText);
            if (!plainText) {
                $('#editPlainText').focus();
                return;
            }
            if (editForm.check().length) return;
            const diacriticText = editForm.code();
            
            // Disable button during submission
            const $btn = $(this);
            const originalBtnText = $btn.html();
            $btn.html('<i class="fas fa-spinner fa-spin"></i> Saving...');
            $btn.prop('disabled', true);
            
            $.ajax({
                url: `${window.DIACRITICAL_ROOT || ''}/api/mappings/${id}`,
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
        
        // Handle enter key in edit modal (in the Choices box it starts a new line)
        $('#editForm').on('keypress', function(e) {
            if (e.which === 13 && e.target.tagName !== 'TEXTAREA') {
                e.preventDefault();
                $('#saveEdit').click();
            }
        });
        
        // Add input event listeners to convert text to lowercase as user types in edit modal
        $('#editPlainText, #editDiacriticText, #editResult, #editAsName, #editChoices').on('input', function() {
            const input = $(this);
            const cursorPosition = this.selectionStart;
            const value = input.val();
            let lowercaseValue = value.toLowerCase();
            if (/plain/i.test(this.id)) {
                lowercaseValue = lowercaseValue.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
            }
            
            // Only update if there's a difference to avoid cursor jumping
            if (value !== lowercaseValue) {
                input.val(lowercaseValue);
                // Restore cursor position
                this.setSelectionRange(cursorPosition, cursorPosition);
            }
        });
        
        // Also convert on paste events in edit modal
        $('#editPlainText, #editDiacriticText, #editResult, #editAsName, #editChoices').on('paste', function() {
            const input = $(this);
            // Use setTimeout to get the value after the paste event completes
            setTimeout(function() {
                input.val(cleanerFor(input[0])(input.val()));
                editForm.update();
            }, 0);
        }).on('blur', function() {
            $(this).val(cleanerFor(this)($(this).val()));
            editForm.update();
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
                url: `${window.DIACRITICAL_ROOT || ''}/api/mappings/${deleteId}`,
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
                url: `${window.DIACRITICAL_ROOT || ''}/api/mappings/batch-delete`,
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
                url: `${window.DIACRITICAL_ROOT || ''}/api/mappings/upload`,
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
            addForm = TypeForm('add', '#diacriticText', '#plainText');
            editForm = TypeForm('edit', '#editDiacriticText', '#editPlainText');
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