const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const labelElem = document.getElementById("label");
const confElem = document.getElementById("confidence");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");

let detecting = false;
let intervalId = null;
let stream = null;

// Add loading indicator
const videoWrapper = document.querySelector('.video-wrapper');
const loadingIndicator = document.createElement('div');
loadingIndicator.className = 'loading-indicator';
loadingIndicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i><p>Initializing camera...</p>';
videoWrapper.appendChild(loadingIndicator);

async function startCamera() {
    try {
        loadingIndicator.style.display = 'flex';
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            } 
        });
        video.srcObject = stream;
        await video.play();
        loadingIndicator.style.display = 'none';
    } catch (err) {
        showNotification("Camera access denied: " + err.message, "error");
        loadingIndicator.style.display = 'none';
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
}

function showNotification(message, type = "info") {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
            <p>${message}</p>
        </div>
        <button class="notification-close"><i class="fas fa-times"></i></button>
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        dismissNotification(notification);
    }, 5000);
    
    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        dismissNotification(notification);
    });
}

function dismissNotification(notification) {
    notification.classList.remove('show');
    setTimeout(() => {
        notification.remove();
    }, 300);
}

async function captureAndPredict() {
    if (!detecting) return;

    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData, session_id: "webcam-session" })
        });

        const data = await response.json();
        const output = document.querySelector(".output");
        
        // Remove previous animation classes
        output.classList.remove("fade-in");
        void output.offsetWidth; // Trigger reflow to restart animation
        
        if (data.label && data.label !== "No Hand Detected") {
            labelElem.textContent = data.label;
            confElem.textContent = (data.confidence * 100).toFixed(1) + "%";
            output.style.borderLeftColor = "var(--success)";
            output.style.background = "rgba(16, 185, 129, 0.1)";
            output.classList.remove("error");
            output.classList.add("fade-in");
        } else {
            labelElem.textContent = "No Hand Detected";
            confElem.textContent = "---";
            output.style.borderLeftColor = "var(--error)";
            output.style.background = "rgba(239, 68, 68, 0.1)";
            output.classList.add("error");
            output.classList.add("fade-in");
        }
    } catch (err) {
        labelElem.textContent = "Error";
        confElem.textContent = err.message;
        const output = document.querySelector(".output");
        output.style.borderLeftColor = "var(--error)";
        output.style.background = "rgba(239, 68, 68, 0.1)";
        output.classList.add("error");
        output.classList.add("fade-in");
    }
}

function startDetection() {
    detecting = true;
    startBtn.style.display = 'none';
    stopBtn.style.display = 'inline-flex';
    intervalId = setInterval(captureAndPredict, 1000);
    showNotification("Detection started", "info");
}

function stopDetection() {
    detecting = false;
    clearInterval(intervalId);
    startBtn.style.display = 'inline-flex';
    stopBtn.style.display = 'none';
    showNotification("Detection stopped", "info");
}

// Event listeners
startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);

// Handle page visibility changes to conserve resources
document.addEventListener('visibilitychange', () => {
    if (document.hidden && detecting) {
        stopDetection();
        showNotification("Detection paused because page is not visible", "info");
    }
});

// Initialize
startCamera();