// DemogFace AI - Premium Frontend Logic

// State variables
let activeTab = 'upload';
let webcamStream = null;
let sessionFaces = [];
let ageChartInstance = null;
let genderChartInstance = null;

// Age groups buckets
const AGE_GROUPS = {
    '0-12 (Child)': { min: 0, max: 12, count: 0 },
    '13-19 (Teen)': { min: 13, max: 19, count: 0 },
    '20-29 (Youth)': { min: 20, max: 29, count: 0 },
    '30-45 (Adult)': { min: 30, max: 45, count: 0 },
    '46-60 (Middle)': { min: 46, max: 60, count: 0 },
    '60+ (Senior)': { min: 61, max: 120, count: 0 }
};

const GENDER_COUNTS = {
    'Male': 0,
    'Female': 0
};

// Initialize elements on load
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    checkModelStatus();
    setupUploadHandlers();
    setupWebcamHandlers();
    
    document.getElementById('clear-stats-btn').addEventListener('click', resetSessionStats);
    document.getElementById('export-report-btn').addEventListener('click', exportDemographicReport);
});

// Switch Tabs (Upload vs Camera)
function switchTab(tabId) {
    activeTab = tabId;
    
    // Toggle active buttons
    document.getElementById('tab-upload').classList.toggle('active', tabId === 'upload');
    document.getElementById('tab-camera').classList.toggle('active', tabId === 'camera');
    
    // Toggle active content divisions
    document.getElementById('content-upload').classList.toggle('active', tabId === 'upload');
    document.getElementById('content-camera').classList.toggle('active', tabId === 'camera');
    
    // Stop camera stream if moving away from webcam
    if (tabId !== 'camera' && webcamStream) {
        stopWebcam();
    }
}

// Fetch model configuration/status from API
async function checkModelStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const indicator = document.getElementById('status-indicator');
        const title = document.getElementById('status-title');
        const desc = document.getElementById('status-desc');
        
        if (data.model_trained) {
            indicator.className = 'status-indicator active';
            indicator.style.color = '#10b981'; // Green glow
            title.innerHTML = `Model: ${data.backbone.toUpperCase()} (Trained)`;
            desc.innerHTML = `Val Loss: ${data.metrics.val_loss} | Age MAE: ${data.metrics.val_age_mae}y | Gender Acc: ${data.metrics.val_gender_acc}%`;
        } else {
            indicator.className = 'status-indicator warning';
            indicator.style.color = '#f59e0b'; // Amber glow
            title.innerHTML = `Model: ${data.backbone.toUpperCase()} (Demo Mode)`;
            desc.innerHTML = "Trained model checkpoint not found. Using local face-detector + mock predictions.";
        }
    } catch (e) {
        console.error('Failed to get status:', e);
        document.getElementById('status-title').innerHTML = "Server connection lost";
        document.getElementById('status-desc').innerHTML = "Verify backend is running on port 8000";
    }
}

// Setup Upload drag-and-drop zone handlers
function setupUploadHandlers() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleImageUploads(e.dataTransfer.files);
        }
    });
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleImageUploads(fileInput.files);
        }
    });
}

// Handle selected file uploads
async function handleImageUploads(files) {
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = ''; // Clear previous preview list
    
    for (let file of files) {
        if (!file.type.startsWith('image/')) continue;
        
        // Add to UI listing
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <div class="file-info">
                <i class="fa-solid fa-image"></i>
                <div class="file-name" title="${file.name}">${file.name}</div>
            </div>
            <div class="file-status pending" id="status-${cleanId(file.name)}">Pending</div>
        `;
        fileList.appendChild(item);
        
        // Send file to FastAPI predict
        try {
            const statusEl = document.getElementById(`status-${cleanId(file.name)}`);
            statusEl.textContent = 'Uploading...';
            
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Prediction API failed');
            
            const data = await response.json();
            
            statusEl.textContent = `${data.faces_detected} detected`;
            statusEl.className = 'file-status success';
            
            // Show annotated image in view
            displayAnnotatedImage(data.annotated_image);
            
            // Append detections to session state
            processDetections(data.predictions);
            
        } catch (err) {
            console.error(err);
            const statusEl = document.getElementById(`status-${cleanId(file.name)}`);
            statusEl.textContent = 'Error';
            statusEl.className = 'file-status error';
        }
    }
}

// Clean filename string for HTML ID
function cleanId(name) {
    return name.replace(/[^a-zA-Z0-9]/g, '_');
}

// Webcam stream controls
function setupWebcamHandlers() {
    const toggleBtn = document.getElementById('webcam-toggle-btn');
    const captureBtn = document.getElementById('capture-btn');
    
    toggleBtn.addEventListener('click', async () => {
        if (webcamStream) {
            stopWebcam();
        } else {
            await startWebcam();
        }
    });
    
    captureBtn.addEventListener('click', captureFrame);
}

async function startWebcam() {
    const video = document.getElementById('webcam-stream');
    const toggleBtn = document.getElementById('webcam-toggle-btn');
    const captureBtn = document.getElementById('capture-btn');
    const wrapper = document.querySelector('.camera-stream-wrapper');
    
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });
        
        video.srcObject = webcamStream;
        wrapper.classList.add('streaming');
        captureBtn.disabled = false;
        toggleBtn.innerHTML = '<i class="fa-solid fa-power-off"></i> Stop Stream';
        toggleBtn.style.background = 'rgba(239, 68, 68, 0.2)';
        toggleBtn.style.borderColor = 'rgba(239, 68, 68, 0.4)';
    } catch (e) {
        console.error('Webcam streaming error:', e);
        alert('Could not access webcam stream. Ensure camera permissions are granted.');
    }
}

function stopWebcam() {
    const video = document.getElementById('webcam-stream');
    const toggleBtn = document.getElementById('webcam-toggle-btn');
    const captureBtn = document.getElementById('capture-btn');
    const wrapper = document.querySelector('.camera-stream-wrapper');
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    video.srcObject = null;
    wrapper.classList.remove('streaming');
    captureBtn.disabled = true;
    toggleBtn.innerHTML = '<i class="fa-solid fa-power-off"></i> Start Stream';
    toggleBtn.style.background = '';
    toggleBtn.style.borderColor = '';
}

// Capture current webcam frame and send to backend
function captureFrame() {
    const video = document.getElementById('webcam-stream');
    const canvas = document.getElementById('photo-canvas');
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to hidden canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to blob and send to backend
    canvas.toBlob(async (blob) => {
        const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
        
        // Add to UI listing
        const fileList = document.getElementById('file-list');
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <div class="file-info">
                <i class="fa-solid fa-camera"></i>
                <div class="file-name">Webcam Capture</div>
            </div>
            <div class="file-status pending" id="status-webcam">Analyzing...</div>
        `;
        fileList.prepend(item);
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Prediction API failed');
            
            const data = await response.json();
            
            const statusEl = document.getElementById('status-webcam');
            statusEl.textContent = `${data.faces_detected} detected`;
            statusEl.className = 'file-status success';
            
            displayAnnotatedImage(data.annotated_image);
            processDetections(data.predictions);
            
        } catch (err) {
            console.error(err);
            const statusEl = document.getElementById('status-webcam');
            statusEl.textContent = 'Error';
            statusEl.className = 'file-status error';
        }
    }, 'image/jpeg');
}

// Display annotated image with drawn faces
function displayAnnotatedImage(base64Image) {
    const placeholder = document.getElementById('placeholder-text');
    const imgView = document.getElementById('annotated-image-view');
    
    placeholder.style.display = 'none';
    imgView.style.display = 'block';
    imgView.src = base64Image;
}

// Update state and UI list with predictions
function processDetections(predictions) {
    const facesList = document.getElementById('detected-faces-list');
    
    if (sessionFaces.length === 0 && predictions.length > 0) {
        facesList.innerHTML = ''; // Clear empty indicator
    }
    
    predictions.forEach(face => {
        // Save to session state list
        sessionFaces.push(face);
        
        // Render item card
        const card = document.createElement('div');
        card.className = 'face-item-card';
        
        const isMale = face.gender === 'Male';
        const genderClass = isMale ? 'male' : 'female';
        const icon = isMale ? 'fa-mars' : 'fa-venus';
        
        card.innerHTML = `
            <div class="face-gender-badge ${genderClass}">
                <i class="fa-solid ${icon}"></i>
            </div>
            <div class="face-details-info">
                <div class="face-row-1">
                    <span class="face-age-span">Age: ${Math.round(face.age)} years</span>
                    <span class="face-gender-span ${genderClass}">${face.gender}</span>
                </div>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-outer">
                        <div class="confidence-bar-inner" style="width: ${face.gender_confidence}%"></div>
                    </div>
                    <span class="confidence-text">${Math.round(face.gender_confidence)}%</span>
                </div>
            </div>
        `;
        facesList.prepend(card);
        
        // Update aggregated stat numbers
        updateAggregatedStats(face);
    });
    
    // Enable export button if there are items in the report
    if (sessionFaces.length > 0) {
        document.getElementById('export-report-btn').disabled = false;
    }
}

// Update aggregate stats calculations
function updateAggregatedStats(face) {
    // 1. Total analyzed
    document.getElementById('stat-total-faces').textContent = sessionFaces.length;
    
    // 2. Average age
    const totalAge = sessionFaces.reduce((sum, f) => sum + f.age, 0);
    const avgAge = totalAge / sessionFaces.length;
    document.getElementById('stat-avg-age').textContent = `${Math.round(avgAge)} y`;
    
    // 3. Gender break down
    if (face.gender in GENDER_COUNTS) {
        GENDER_COUNTS[face.gender]++;
    }
    
    // 4. Age buckets
    for (let groupName in AGE_GROUPS) {
        const group = AGE_GROUPS[groupName];
        if (face.age >= group.min && face.age <= group.max) {
            group.count++;
            break;
        }
    }
    
    updateChartsData();
}

// Reset entire session variables
function resetSessionStats() {
    sessionFaces = [];
    
    // Reset counter maps
    GENDER_COUNTS['Male'] = 0;
    GENDER_COUNTS['Female'] = 0;
    
    for (let groupName in AGE_GROUPS) {
        AGE_GROUPS[groupName].count = 0;
    }
    
    // Reset labels
    document.getElementById('stat-total-faces').textContent = '0';
    document.getElementById('stat-avg-age').textContent = 'N/A';
    document.getElementById('detected-faces-list').innerHTML = `
        <div class="empty-list-text">No faces detected yet</div>
    `;
    
    // Reset display
    document.getElementById('annotated-image-view').style.display = 'none';
    document.getElementById('placeholder-text').style.display = 'flex';
    
    document.getElementById('export-report-btn').disabled = true;
    
    updateChartsData();
}

// Chart.js init system
function initCharts() {
    // 1. Age groups bar chart
    const ageCtx = document.getElementById('age-chart').getContext('2d');
    ageChartInstance = new Chart(ageCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(AGE_GROUPS),
            datasets: [{
                label: 'Detections',
                data: Object.values(AGE_GROUPS).map(g => g.count),
                backgroundColor: 'rgba(79, 70, 229, 0.65)',
                borderColor: '#4f46e5',
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#9ca3af',
                        stepSize: 1,
                        font: { family: 'Inter' }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    ticks: {
                        color: '#9ca3af',
                        font: { family: 'Inter', size: 10 }
                    },
                    grid: { display: false }
                }
            }
        }
    });

    // 2. Gender breakdown pie chart
    const genderCtx = document.getElementById('gender-chart').getContext('2d');
    genderChartInstance = new Chart(genderCtx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(GENDER_COUNTS),
            datasets: [{
                data: Object.values(GENDER_COUNTS),
                backgroundColor: ['#3b82f6', '#ec4899'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#9ca3af',
                        font: { family: 'Inter', size: 12 },
                        padding: 15
                    }
                }
            },
            cutout: '65%'
        }
    });
}

// Update charts datasets
function updateChartsData() {
    if (ageChartInstance) {
        ageChartInstance.data.datasets[0].data = Object.values(AGE_GROUPS).map(g => g.count);
        ageChartInstance.update();
    }
    
    if (genderChartInstance) {
        genderChartInstance.data.datasets[0].data = Object.values(GENDER_COUNTS);
        genderChartInstance.update();
    }
}

// Export Session Demographics data
function exportDemographicReport() {
    if (sessionFaces.length === 0) return;
    
    const totalAge = sessionFaces.reduce((sum, f) => sum + f.age, 0);
    const avgAge = (totalAge / sessionFaces.length).toFixed(1);
    
    // Generate Text Report Content
    let reportText = `==================================================\n`;
    reportText += `   DEMOGFACE AI - DEMOGRAPHIC ANALYTICS REPORT\n`;
    reportText += `   Generated: ${new Date().toLocaleString()}\n`;
    reportText += `==================================================\n\n`;
    
    reportText += `SUMMARY STATS:\n`;
    reportText += `--------------------------------------------------\n`;
    reportText += `Total Faces Analyzed: ${sessionFaces.length}\n`;
    reportText += `Average Estimated Age: ${avgAge} years\n\n`;
    
    reportText += `GENDER BREAKDOWN:\n`;
    reportText += `--------------------------------------------------\n`;
    const mCount = GENDER_COUNTS['Male'];
    const fCount = GENDER_COUNTS['Female'];
    const mPercent = ((mCount / sessionFaces.length) * 100).toFixed(1);
    const fPercent = ((fCount / sessionFaces.length) * 100).toFixed(1);
    reportText += `- Male: ${mCount} (${mPercent}%)\n`;
    reportText += `- Female: ${fCount} (${fPercent}%)\n\n`;
    
    reportText += `AGE DISTRIBUTION:\n`;
    reportText += `--------------------------------------------------\n`;
    for (let groupName in AGE_GROUPS) {
        const count = AGE_GROUPS[groupName].count;
        const percent = ((count / sessionFaces.length) * 100).toFixed(1);
        reportText += `- ${groupName}: ${count} (${percent}%)\n`;
    }
    
    reportText += `\n==================================================\n`;
    reportText += `   DETAILED CAPTURES (${sessionFaces.length} records):\n`;
    reportText += `==================================================\n`;
    sessionFaces.forEach((f, idx) => {
        reportText += `Face #${idx+1}: Gender=${f.gender} (Conf=${f.gender_confidence}%), Age=${Math.round(f.age)}y\n`;
    });
    
    // Create download element
    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Demographics_Report_${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
