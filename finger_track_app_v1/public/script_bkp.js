// Improved script.js
// Requirements: index.html must include MediaPipe scripts:
// <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
// <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
// <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>

const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');

const statusEl = document.getElementById('status');
const handedEl = document.getElementById('handedness');
const countEl = document.getElementById('count');

// Make sure the canvas pixel size exactly matches the video frame size to avoid misalignment
function ensureCanvasSize(image) {
  const w = image && image.width ? image.width : videoElement.videoWidth;
  const h = image && image.height ? image.height : videoElement.videoHeight;
  if (w && h && (canvasElement.width !== w || canvasElement.height !== h)) {
    canvasElement.width = w;
    canvasElement.height = h;
    canvasElement.style.width = '100%';
    canvasElement.style.height = '100%';
  }
}

// Angle at point B between A-B and C-B (degrees)
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

// Analyze a single hand's landmarks; returns count and which fingers are extended
function analyzeHand(landmarks) {
  // indices: 0 wrist
  // thumb: 2(MCP),3(IP),4(TIP)
  // index: 5(MCP),6(PIP),8(TIP)
  // middle: 9,10,12
  // ring: 13,14,16
  // pinky: 17,18,20
  const extended = [false, false, false, false, false]; // thumb, index, middle, ring, pinky

  // Thumb: angle at IP (3)
  try {
    const thumbAng = angleBetweenPoints(landmarks[2], landmarks[3], landmarks[4]);
    extended[0] = thumbAng > 150; // thumb extended -> near straight
  } catch (e) {
    extended[0] = false;
  }

  // Other fingers: angle at PIP; if near 180 => extended
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
      extended[i + 1] = ang > 160; // tuned threshold
    } catch (e) {
      extended[i + 1] = false;
    }
  }

  const count = extended.reduce((s, v) => s + (v ? 1 : 0), 0);
  return { count, extended };
}

// simple mode function
function mode(arr) {
  const counts = {};
  for (const v of arr) counts[v] = (counts[v] || 0) + 1;
  let best = null;
  for (const k of Object.keys(counts)) {
    if (best === null || counts[k] > counts[best]) best = k;
  }
  return best;
}

let countHistory = [];
let handedHistory = [];
const HISTORY_MAX = 6;

// Called by MediaPipe on each processed frame
function onResults(results) {
  // ensure canvas matches frame size
  ensureCanvasSize(results.image);

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  // draw camera frame (pixel-aligned)
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    // draw all hands clearly
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      drawConnectors(canvasCtx, results.multiHandLandmarks[i], HAND_CONNECTIONS, { color: '#00e0a8', lineWidth: 5 });
      drawLandmarks(canvasCtx, results.multiHandLandmarks[i], { color: '#ffffff', lineWidth: 2, radius: 4 });
    }

    // primary = first detected hand
    const primaryLandmarks = results.multiHandLandmarks[0];
    let handednessLabel = (results.multiHandedness && results.multiHandedness[0] && results.multiHandedness[0].label) || 'Unknown';

    // If using a front-facing camera (selfie), invert the label for visual match
    // because MediaPipe returns handedness from subject's POV
    const usingSelfieCam = true; // set to false if using rear camera
    if (usingSelfieCam) {
    if (handednessLabel === 'Left') handednessLabel = 'Right';
    else if (handednessLabel === 'Right') handednessLabel = 'Left';
    }


    // analyze finger angles
    const analysis = analyzeHand(primaryLandmarks);

    // smoothing / history
    countHistory.push(analysis.count);
    if (countHistory.length > HISTORY_MAX) countHistory.shift();
    handedHistory.push(handednessLabel);
    if (handedHistory.length > HISTORY_MAX) handedHistory.shift();

    const displayedCount = Number(mode(countHistory));
    const displayedHand = mode(handedHistory);

    // UI updates
    statusEl.innerText = 'Hand detected';
    countEl.innerText = `Fingers: ${displayedCount}`;
    handedEl.innerText = `Hand: ${displayedHand}`;

    // highlight the fingertip circles (visible + different if extended)
    const tipIndices = [4, 8, 12, 16, 20];
    for (let t = 0; t < tipIndices.length; t++) {
      const lm = primaryLandmarks[tipIndices[t]];
      const x = lm.x * canvasElement.width;
      const y = lm.y * canvasElement.height;

      canvasCtx.beginPath();
      canvasCtx.arc(x, y, 10, 0, Math.PI * 2);
      canvasCtx.fillStyle = analysis.extended[t] ? 'rgba(0,224,168,0.95)' : 'rgba(255,255,255,0.12)';
      canvasCtx.fill();

      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = 'rgba(0,0,0,0.6)';
      canvasCtx.stroke();
    }

    // large overlay (top-left), pixel-aligned
    canvasCtx.font = 'bold 48px Inter, Arial';
    canvasCtx.fillStyle = 'rgba(0,224,168,0.95)';
    canvasCtx.textAlign = 'left';
    canvasCtx.fillText(String(displayedCount), 18, 52);
    canvasCtx.font = '16px Inter, Arial';
    canvasCtx.fillStyle = 'rgba(255,255,255,0.95)';
    canvasCtx.fillText(displayedHand, 18, 76);

  } else {
    // no hand
    statusEl.innerText = 'No hands detected';
    countHistory = [];
    handedHistory = [];
    countEl.innerText = 'Fingers: 0';
    handedEl.innerText = 'Hand: —';
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

// Use MediaPipe Camera helper (starts stream and calls send())
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 1280,
  height: 720
});

async function startCamera() {
  try {
    await camera.start();
    statusEl.innerText = 'Camera started. Show your hand to count fingers.';
  } catch (err) {
    statusEl.innerText = 'Camera start failed: ' + (err && err.message ? err.message : err);
    console.error(err);
  }
}

// run
startCamera();
