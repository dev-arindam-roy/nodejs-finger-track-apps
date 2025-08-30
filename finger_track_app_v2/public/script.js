// script.js
// Uses TensorFlow.js hand-pose-detection model (MediaPipeHands backend).
// Features:
// - Angle-based finger detection (3D angles).
// - Sliding-window median + exponential smoothing.
// - On-screen toggles for skeleton, multi-hand, mirror video.

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const statusEl = document.getElementById('status');
const handedEl = document.getElementById('handedness');
const countEl = document.getElementById('count');

const btnSkeleton = document.getElementById('toggle-skeleton');
const btnMulti = document.getElementById('toggle-multihand');
const btnMirror = document.getElementById('toggle-mirror');

let showSkeleton = true;
let multiHand = false;
let mirror = false;

btnSkeleton.addEventListener('click', () => {
  showSkeleton = !showSkeleton;
  btnSkeleton.innerText = showSkeleton ? 'Hide Skeleton' : 'Show Skeleton';
});
btnMulti.addEventListener('click', () => {
  multiHand = !multiHand;
  btnMulti.innerText = `Multi-hand: ${multiHand ? 'On' : 'Off'}`;
});
btnMirror.addEventListener('click', () => {
  mirror = !mirror;
  btnMirror.innerText = `Mirror: ${mirror ? 'On' : 'Off'}`;
  // mirror video visually (canvas drawing will flip)
  video.style.transform = mirror ? 'scaleX(-1)' : 'none';
});

// canvas sizing helper
function fitCanvasToVideo() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;
  canvas.width = vw;
  canvas.height = vh;
  canvas.style.width = video.clientWidth + 'px';
  canvas.style.height = video.clientHeight + 'px';
}

// ---- angle helper (3D) ----
function angleBetween(a, b, c) {
  // returns degrees between BA and BC at point B
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

// ---- analyze single hand landmarks ----
function analyzeLandmarks(landmarks) {
  // landmarks are array of 21 points (x,y,z) normalized [0..1]
  // We'll compute whether each finger is extended using angle thresholds.
  // indices: thumb [1..4], index [5..8], middle [9..12], ring [13..16], pinky [17..20]
  const extended = [false, false, false, false, false];

  // Thumb: measure angle at CMC/MCP (use 2-3-4)
  try {
    const thumbAngle = angleBetween(landmarks[2], landmarks[3], landmarks[4]);
    extended[0] = thumbAngle > 150; // near straight
  } catch (e) {
    extended[0] = false;
  }

  // Other fingers: angle at PIP joint (mcp-pip-tip)
  const fingers = [
    [5,6,8],
    [9,10,12],
    [13,14,16],
    [17,18,20]
  ];
  for (let i=0;i<fingers.length;i++){
    const [mcp,pip,tip] = fingers[i];
    try {
      const ang = angleBetween(landmarks[mcp], landmarks[pip], landmarks[tip]);
      extended[i+1] = ang > 160; // tuned threshold
    } catch (e) {
      extended[i+1] = false;
    }
  }

  const count = extended.reduce((s,v)=>s+(v?1:0),0);
  return {extended, count};
}

// ---- smoothing / history ----
const COUNT_HISTORY = 8;
const HAND_HISTORY = 8;
let countHistory = []; // array of integers
let handHistory = [];  // array of strings (Left/Right/Unknown)
let smoothCount = 0;
let smoothHand = '—';

function pushHistory(count, handLabel) {
  countHistory.push(count);
  if (countHistory.length > COUNT_HISTORY) countHistory.shift();
  handHistory.push(handLabel);
  if (handHistory.length > HAND_HISTORY) handHistory.shift();

  // median of countHistory for debouncing
  const sorted = [...countHistory].sort((a,b)=>a-b);
  const mid = Math.floor(sorted.length/2);
  const median = sorted.length%2===1 ? sorted[mid] : Math.round((sorted[mid-1]+sorted[mid])/2);

  // exponential smoothing
  smoothCount = Math.round((smoothCount*0.6) + (median*0.4));
  // mode for hand
  const freq = {};
  for (const h of handHistory) freq[h] = (freq[h]||0)+1;
  smoothHand = Object.keys(freq).reduce((a,b)=> freq[a]>freq[b]?a:b);
}

// ---- drawing helpers ----
function drawLine(a,b,width=4, color='rgba(0,224,168,0.95)') {
  ctx.lineWidth = width;
  ctx.strokeStyle = color;
  ctx.beginPath();
  ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
  ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
  ctx.stroke();
}
function drawCircle(p,r=6, fill='rgba(0,224,168,0.95)') {
  ctx.beginPath();
  ctx.arc(p.x * canvas.width, p.y * canvas.height, r, 0, Math.PI*2);
  ctx.fillStyle = fill;
  ctx.fill();
}

// ---- finger connections for drawing (index pairs) ----
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // thumb
  [0,5],[5,6],[6,7],[7,8],       // index
  [0,9],[9,10],[10,11],[11,12],  // middle
  [0,13],[13,14],[14,15],[15,16],// ring
  [0,17],[17,18],[18,19],[19,20] // pinky
];

// ---- TF.js Hand Pose Detection setup ----
let detector = null;
let modelLoaded = false;

async function initDetector() {
  // use the hand-pose-detection model (MediaPipeHands)
  const model = handPoseDetection.SupportedModels.MediaPipeHands;

  // config: runtime uses "mediapipe" which leverages the mediapipe wasm model
  // but this package integrates with TF.js; we still satisfy "TensorFlow.js ecosystem"
  const detectorConfig = {
    runtime: 'mediapipe',
    modelType: 'full', // 'lite' or 'full'
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
  };

  detector = await handPoseDetection.createDetector(model, detectorConfig);
  modelLoaded = true;
  statusEl.innerText = 'Model loaded — starting camera...';
  startCamera();
}

// ---- camera start ----
async function startCamera(){
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {width:1280, height:720}, audio:false});
    video.srcObject = stream;
    await video.play();
    fitCanvasToVideo();
    statusEl.innerText = 'Camera started. Show your hand.';
    requestAnimationFrame(processFrame);
  } catch (e) {
    statusEl.innerText = 'Camera error: ' + e.message;
    console.error(e);
  }
}

// ---- main loop ----
async function processFrame() {
  if (!modelLoaded) return requestAnimationFrame(processFrame);
  if (video.readyState < 2) return requestAnimationFrame(processFrame);
  fitCanvasToVideo();

  // detect hands (multiHand or single)
  const maxHands = multiHand ? 2 : 1;
  let hands;
  try {
    hands = await detector.estimateHands(video, {flipHorizontal:false}); // we'll handle mirror manually
  } catch (err) {
    console.error('detect error', err);
    hands = [];
  }

  // render
  ctx.save();
  ctx.clearRect(0,0,canvas.width,canvas.height);

  // if mirror mode, flip canvas horizontally so visuals match selfie
  if (mirror) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1,1);
  }

  // draw camera (video frame) underneath by copying video - faster to not draw video to canvas since <video> is visible, but we want overlays only.
  // we only draw overlays; video element remains visible under the canvas.

  if (hands && hands.length > 0) {
    // clamp to maxHands
    const shownHands = hands.slice(0, maxHands);

    for (let i=0;i<shownHands.length;i++){
      const h = shownHands[i];
      const keypoints = h.keypoints3D || h.keypoints; // keypoints3D might be available
      const landmarks = (keypoints && keypoints.length===21) ? keypoints.map(k => ({x:k.x, y:k.y, z:(k.z||0)})) : null;

      // If keypoints are normalized to video pixel coords, normalize to 0..1:
      let normLandmarks = landmarks;
      if (landmarks && landmarks[0] && landmarks[0].x > 1.5) {
        // coordinates in pixels -> convert back to normalized
        normLandmarks = landmarks.map(pt => ({x: pt.x / canvas.width, y: pt.y / canvas.height, z: pt.z}));
      }

      // analyze only if we have 21 landmarks
      if (!normLandmarks || normLandmarks.length !== 21) continue;
      const analysis = analyzeLandmarks(normLandmarks);

      // detect handedness: model returns h.handedness sometimes in latest API
      let handednessLabel = 'Unknown';
      if (h.handedness && h.handedness.length) handednessLabel = h.handedness[0].label || 'Unknown';
      // When mirror mode ON, flip label so it matches viewer
      if (mirror) {
        if (handednessLabel === 'Left') handednessLabel = 'Right';
        else if (handednessLabel === 'Right') handednessLabel = 'Left';
      }

      // push history only for first shown hand
      if (i===0) pushHistory(analysis.count, handednessLabel);

      // draw skeleton if enabled
      if (showSkeleton) {
        // draw connections
        for (const c of CONNECTIONS) {
          const a = normLandmarks[c[0]], b = normLandmarks[c[1]];
          drawLine(a, b, 3, 'rgba(0,224,168,0.95)');
        }
        // draw points
        for (let pi=0; pi<normLandmarks.length; pi++) {
          const p = normLandmarks[pi];
          const isTip = [4,8,12,16,20].includes(pi);
          drawCircle(p, isTip?6:3, isTip && analysis.extended[[4,8,12,16,20].indexOf(pi)] ? 'rgba(0,224,168,0.95)' : 'rgba(255,255,255,0.12)');
        }
      } else {
        // draw only highlight tips
        for (const tidx of [4,8,12,16,20]) {
          const p = normLandmarks[tidx];
          const idx = [4,8,12,16,20].indexOf(tidx);
          const fill = analysis.extended[idx] ? 'rgba(0,224,168,0.95)' : 'rgba(255,255,255,0.08)';
          drawCircle(p, 7, fill);
        }
      }
    }

    // update UI from smoothed values
    countEl.innerText = `Fingers: ${smoothCount}`;
    handedEl.innerText = `Hand: ${smoothHand}`;
    statusEl.innerText = `Detected ${hands.length} hand(s)`;
  } else {
    // no hands
    statusEl.innerText = 'No hands detected';
    countEl.innerText = 'Fingers: 0';
    handedEl.innerText = 'Hand: —';
    countHistory = []; handHistory = []; smoothCount = 0; smoothHand = '—';
  }

  ctx.restore();
  requestAnimationFrame(processFrame);
}

// ---- init ----
statusEl.innerText = 'Loading TF detector...';
initDetector().catch(e => {
  statusEl.innerText = 'Model init failed: ' + e.message;
  console.error(e);
});
