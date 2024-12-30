// Access the user's camera
const video = document.getElementById('video');
const capturedImage = document.getElementById('captured-image');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const retakeButton = document.getElementById('retake');
const submitButton = document.getElementById('submit');

// Start video stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing camera: ', err);
    });

canvas.width = 640;
canvas.height = 480;

// Capture button logic
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height); // Draw scaled image

    // Display captured image
    const imageData = canvas.toDataURL('image/png'); // Base64 encode
    capturedImage.src = imageData;
    capturedImage.style.display = 'block';
    video.style.display = 'none';
    captureButton.style.display = 'none';
    retakeButton.style.display = 'block';
    submitButton.style.display = 'block';
});

// Retake button logic
retakeButton.addEventListener('click', () => {
    capturedImage.style.display = 'none';
    video.style.display = 'block';
    captureButton.style.display = 'block';
    retakeButton.style.display = 'none';
    submitButton.style.display = 'none';
});

// Submit button logic (direct submission without a separate form)
submitButton.addEventListener('click', () => {
    const imageData = canvas.toDataURL('image/png');

    // Resize the image (resize to 300px width while maintaining aspect ratio)
    const img = new Image();
    img.src = imageData;
    img.onload = function() {
        const resizedCanvas = document.createElement('canvas');
        const resizedContext = resizedCanvas.getContext('2d');
        const width = 300; // Resize to 300px width
        const height = (img.height / img.width) * width; // Maintain aspect ratio
        resizedCanvas.width = width;
        resizedCanvas.height = height;
        resizedContext.drawImage(img, 0, 0, width, height);

        // Convert resized image to Base64
        const resizedImageData = resizedCanvas.toDataURL('image/png');

        // Create a form dynamically and submit
        const form = document.createElement('form');
        form.action = '/upload';
        form.method = 'POST';
        form.enctype = 'multipart/form-data';

        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'photo-data';
        input.value = resizedImageData;
        form.appendChild(input);

        document.body.appendChild(form);
        form.submit();
    };
});
