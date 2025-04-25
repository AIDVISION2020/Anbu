import React, { useRef, useState } from 'react';
import { detectImage } from './api.js';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [detections, setDetections] = useState([]);
  const [capturedImage, setCapturedImage] = useState(null);
  const [uploadCanvasVisible, setUploadCanvasVisible] = useState(false);
  const [cameraCanvasVisible, setCameraCanvasVisible] = useState(false);
  const [showCameraSection, setShowCameraSection] = useState(false);
  const [stream, setStream] = useState(null);

  const videoRef = useRef(null);
  const uploadCanvasRef = useRef(null);
  const cameraCanvasRef = useRef(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadCanvasVisible(false);
    setCameraCanvasVisible(false);
  };

  const handleDetect = async () => {
    if (!selectedFile) return;
    const result = await detectImage(selectedFile);
    if (result) {
      setDetections(result.detections);
      setCapturedImage(null);
      setUploadCanvasVisible(true);
      setCameraCanvasVisible(false);
      drawBoundingBoxes(result.detections, URL.createObjectURL(selectedFile), uploadCanvasRef);
    }
  };

  const drawBoundingBoxes = (detections, fileURL, canvasReference) => {
    const img = new Image();
    img.src = fileURL;
    img.onload = () => {
      const canvas = canvasReference.current;
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        ctx.strokeStyle = 'lime';
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.font = '16px Arial';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        const text = `${det.label} (${(det.confidence * 100).toFixed(1)}%)`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);

        ctx.fillStyle = 'yellow';
        ctx.fillText(text, x1 + 5, y1 - 7);
      });
    };
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = mediaStream;
      setStream(mediaStream);
      setCapturedImage(null);
      setUploadCanvasVisible(false);
      setCameraCanvasVisible(false);
    } catch (err) {
      console.error('Camera error:', err);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setStream(null);
    }
    setCapturedImage(null);
    setCameraCanvasVisible(false);
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
      const url = URL.createObjectURL(file);
      const result = await detectImage(file);
      if (result) {
        stopCamera();
        setCapturedImage(url);
        setDetections(result.detections);
        setCameraCanvasVisible(true);
        setUploadCanvasVisible(false);

        setTimeout(() => {
          drawBoundingBoxes(result.detections, url, cameraCanvasRef);
        }, 100);
      }
    }, 'image/jpeg');
  };

  const toggleCameraSection = () => {
    setShowCameraSection((prev) => {
      if (!prev) {
        startCamera();
      } else {
        stopCamera();
      }
      return !prev;
    });
  };

  const startDetection = () => {
    stopCamera();
    startCamera();
  };

  return (
    <div className="App" style={{ fontFamily: 'Arial, sans-serif', textAlign: 'center' }}>
      <h1>YOLOv8 Object Detection (React + FastAPI)</h1>

      <div style={{
        display: 'flex',
        justifyContent: 'space-around',
        alignItems: 'flex-start',
        flexWrap: 'wrap',
        gap: '20px'
      }}>
        {/* Left: Camera Section */}
        <div style={{ flex: '1', minWidth: '300px' }}>
          <button onClick={toggleCameraSection} style={{ marginBottom: '10px' }}>
            {showCameraSection ? 'Hide Live Camera' : 'Show Live Camera'}
          </button>

          {showCameraSection && (
            <>
              {cameraCanvasVisible ? (
                <canvas
                  ref={cameraCanvasRef}
                  style={{ width: '100%', maxWidth: '400px', height: '300px', objectFit: 'contain', border: '2px solid black' }}
                />
              ) : (
                <video
                  ref={videoRef}
                  autoPlay
                  style={{ width: '100%', maxWidth: '400px', height: '300px', border: '2px solid black' }}
                />
              )}
              <br />
              <button onClick={captureImage}>Capture & Detect</button>
              <button onClick={startDetection}>Start Camera</button> {/* Added Start Button */}
              <button onClick={stopCamera}>Stop Camera</button>
            </>
          )}
        </div>

        {/* Right: Upload Section */}
        <div style={{ flex: '1', minWidth: '300px' }}>
          <input type="file" onChange={handleFileChange} />
          <br />
          <button onClick={handleDetect}>Upload & Detect</button>
        </div>
      </div>

      {/* Detection Canvas for Uploaded Image */}
      {uploadCanvasVisible && (
        <div style={{ marginTop: '30px' }}>
          <canvas
            ref={uploadCanvasRef}
            style={{ border: '2px solid #333', width: '90%', maxWidth: '500px' }}
          />
        </div>
      )}

      {/* Detected Objects Table */}
      {detections.length > 0 && (
        <div style={{ marginTop: '30px' }}>
          <h3>Detected Objects</h3>
          <table style={{
            margin: '0 auto',
            borderCollapse: 'collapse',
            width: '90%',
            maxWidth: '600px'
          }}>
            <thead>
              <tr>
                <th style={{ border: '1px solid black', padding: '8px' }}>Label</th>
                <th style={{ border: '1px solid black', padding: '8px' }}>Confidence</th>
                <th style={{ border: '1px solid black', padding: '8px' }}>Bounding Box (x1, y1, x2, y2)</th>
              </tr>
            </thead>
            <tbody>
              {detections.map((det, index) => {
                const [x1, y1, x2, y2] = det.bbox;
                return (
                  <tr key={index}>
                    <td style={{ border: '1px solid black', padding: '8px' }}>{det.label}</td>
                    <td style={{ border: '1px solid black', padding: '8px' }}>{(det.confidence * 100).toFixed(1)}%</td>
                    <td style={{ border: '1px solid black', padding: '8px' }}>{`${x1}, ${y1}, ${x2}, ${y2}`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
