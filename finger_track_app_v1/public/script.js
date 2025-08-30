// Finger Tracking with Smoothing, Robust Angles, and Overlay Toggles

const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');

const statusEl = document.getElementById('status');
const handedEl = document.getElementById('handedness');
const countEl = document.getElementById('count');

let showSkeleton = true;
let showMultipleHands = false;

// ----- Utility: Ensure canvas matches video size -----
function ensureCanvasSize(image) {
  const w = image?.width || videoElement.videoWidth;
  const h = image?.height || videoElement.videoHeight;
  if (w && h && (canvasElement.width !== w || canvasElement.height !== h)) {
    canvasElement.width = w;
    canvasElement.height = h;
    canvasElement.style.width = '100%';
    canvasElement.style.height = '100%';
  }
}

// ----- Angle between points (3D) -----
function angleBetweenPoints(a, b, c) {
  const abx = a.x - b.x, aby = a.y - b.y, abz = (a.z || 0) - (b.z || 0);
  const cbx = c.x - b.x, cby = c.y - b.y, cbz = (c.z || 0) - (b.z || 0);
  const dot = abx * cbx + aby * cby + abz * cbz;
  const mag1 = Math.hypot(abx, aby, abz);
  const mag2 = Math.hypot(cbx, cby, cbz);
  if (mag1 === 0 || mag2 === 0) return 0;
  let cos = dot / (mag1 * mag2);
  cos = Math.max(-1, Math.min(1, cos));
  return Math.acos(cos) * (180 / Math.PI);
}

// ----- Analyze fingers with robust angle detection -----
function analyzeHand(landmarks) {
  const extended = [false, false, false, false, false]; // thumb, index, middle, ring, pinky

  // Thumb: angle at MCP (2) or IP (3)
  try {
    const thumbAng = angleBetweenPoints(landmarks[2], landmarks[3], landmarks[4]);
    extended[0] = thumbAng > 150; // extended if nearly straight
  } catch (e) {
    extended[0] = false;
  }

  // Other fingers: PIP joint angles
  const fingers = [
    [5, 6, 8],
    [9, 10, 12],
    [13, 14, 16],
    [17, 18, 20]
  ];
  for (let i = 0; i < fingers.length; i++) {
    const [mcp, pip, tip] = fingers[i];
    try {
      const ang = angleBetweenPoints(landmarks[mcp], landmarks[pip], landmarks[tip]);
      extended[i + 1] = ang > 160; // extended if almost straight
    } catch (e) {
      extended[i + 1] = false;
    }
  }

  const count = extended.reduce((sum, v) => sum + (v ? 1 : 0), 0);
  return { count, extended };
}

// ----- Mode & smoothing helpers -----
function mode(arr) {
  const counts = {};
  for (const v of arr) counts[v] = (counts[v] || 0) + 1;
  return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}

let countHistory = [];
let handedHistory = [];
const HISTORY_MAX = 8;

// ----- Main callback -----
function onResults(results) {
  ensureCanvasSize(results.image);

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks?.length > 0) {
    const handsToShow = showMultipleHands
      ? results.multiHandLandmarks.length
      : Math.min(1, results.multiHandLandmarks.length);

    for (let i = 0; i < handsToShow; i++) {
      const landmarks = results.multiHandLandmarks[i];
      let handednessLabel = results.multiHandedness?.[i]?.label || 'Unknown';

      // Flip handedness for selfie camera
      const usingSelfieCam = true;
      if (usingSelfieCam) {
        handednessLabel = handednessLabel === 'Left' ? 'Right'
                         : handednessLabel === 'Right' ? 'Left' : handednessLabel;
      }

      const analysis = analyzeHand(landmarks);

      // Update history for smoothing (only for first hand shown)
      if (i === 0) {
        countHistory.push(analysis.count);
        if (countHistory.length > HISTORY_MAX) countHistory.shift();
        handedHistory.push(handednessLabel);
        if (handedHistory.length > HISTORY_MAX) handedHistory.shift();
      }

      if (showSkeleton) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00e0a8', lineWidth: 4 });
        drawLandmarks(canvasCtx, landmarks, { color: '#ffffff', lineWidth: 2, radius: 4 });
      }

      // Highlight tips if extended
      const tipIndices = [4, 8, 12, 16, 20];
      for (let t = 0; t < tipIndices.length; t++) {
        const lm = landmarks[tipIndices[t]];
        const x = lm.x * canvasElement.width;
        const y = lm.y * canvasElement.height;
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 10, 0, Math.PI * 2);
        canvasCtx.fillStyle = analysis.extended[t]
          ? 'rgba(0,224,168,0.95)'
          : 'rgba(255,255,255,0.1)';
        canvasCtx.fill();
      }
    }

    // Apply smoothing
    const displayedCount = Number(mode(countHistory));
    const displayedHand = mode(handedHistory);

    // UI
    statusEl.innerText = 'Hand detected';
    countEl.innerText = `Fingers: ${displayedCount}`;
    handedEl.innerText = `Hand: ${displayedHand}`;

  } else {
    statusEl.innerText = 'No hands detected';
    countEl.innerText = 'Fingers: 0';
    handedEl.innerText = 'Hand: —';
    countHistory = [];
    handedHistory = [];
  }

  canvasCtx.restore();
}

// ----- MediaPipe Hands setup -----
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.6
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 1280,
  height: 720
});

camera.start().then(() => {
  statusEl.innerText = 'Camera started. Show your hand.';
}).catch(err => {
  statusEl.innerText = 'Camera failed: ' + err.message;
});

// ----- Keyboard controls -----
window.addEventListener('keydown', (e) => {
  if (e.key.toLowerCase() === 's') {
    showSkeleton = !showSkeleton;
    console.log(`Skeleton ${showSkeleton ? 'ON' : 'OFF'}`);
  }
  if (e.key.toLowerCase() === 'm') {
    showMultipleHands = !showMultipleHands;
    console.log(`Multi-hand mode ${showMultipleHands ? 'ON' : 'OFF'}`);
  }
});
