// ===================================FILE UPLOAD=============================
function fileUpload() {
            return {
                isDragging: false,
                file: null,
                isPdf: false,
                handleFileSelect(event) {
                    this.setFile(event.target.files[0]);
                },
                handleDrop(event) {
                    this.isDragging = false;
                    this.setFile(event.dataTransfer.files[0]);
                },
                setFile(file) {
                    if (file && file.type === 'application/pdf') {
                        this.file = file;
                        this.isPdf = true;
                    } else if (file) {
                        this.file = file;
                        this.isPdf = false;
                    }
                },
                reset() {
                    this.file = null;
                    this.isPdf = false;
                    document.getElementById('fileInput').value = '';
                }
            }
        }