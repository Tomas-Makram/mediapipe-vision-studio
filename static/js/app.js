const el = (id) => document.getElementById(id);
const faceSummary = el('faceSummary');
const faceDistance = el('faceDistance');
const facesList = el('facesList');
const unknownSelect = el('unknownSelect');
const handsList = el('handsList');
const peopleList = el('peopleList');
const peopleSelect = el('peopleSelect');
const registerBtn = el('registerBtn');
const personName = el('personName');
const registerMessage = el('registerMessage');
const peopleMessage = el('peopleMessage');
const connectionStatus = el('connectionStatus');
const renameInput = el('renameInput');
const wordInput = el('wordInput');
const wordLayer = el('wordLayer');
const videoWrap = el('videoWrap');
const drawCanvas = el('drawCanvas');
const cursorDot = el('cursorDot');
const gestureMessage = el('gestureMessage');
const gesturesList = el('gesturesList');
const drawColor = el('drawColor');
const performanceText = el('performanceText');

let currentState = null;
let mode = 'draw';
let words = [];
let grabbedWordId = null;
let lastPinch = false;
let lastDrawPoint = null;
let ctx = drawCanvas.getContext('2d');

function setMessage(node, text) { node.textContent = text || ''; }
function selectedUnknownId() { return unknownSelect.value || ''; }
function selectedPersonName() { return peopleSelect.value || ''; }

function resizeCanvas() {
  const rect = videoWrap.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  drawCanvas.width = Math.round(rect.width * ratio);
  drawCanvas.height = Math.round(rect.height * ratio);
  drawCanvas.style.width = rect.width + 'px';
  drawCanvas.style.height = rect.height + 'px';
  ctx = drawCanvas.getContext('2d');
  ctx.scale(ratio, ratio);
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = drawColor ? drawColor.value : '#ffffff';
}

function renderPeople(people) {
  const selected = peopleSelect.value;
  peopleList.innerHTML = '';
  peopleSelect.innerHTML = '';
  if (!people || people.length === 0) {
    peopleList.innerHTML = '<li>No registered people yet</li>';
    peopleSelect.innerHTML = '<option value="">No people</option>';
    return;
  }
  for (const p of people) {
    const li = document.createElement('li');
    li.textContent = `${p.name} - samples: ${p.samples}`;
    peopleList.appendChild(li);
    const opt = document.createElement('option');
    opt.value = p.name;
    opt.textContent = `${p.name} (${p.samples})`;
    peopleSelect.appendChild(opt);
  }
  if ([...peopleSelect.options].some(o => o.value === selected)) peopleSelect.value = selected;
}

function renderUnknowns(unknowns) {
  const selected = unknownSelect.value;
  unknownSelect.innerHTML = '';
  if (!unknowns || unknowns.length === 0) {
    unknownSelect.innerHTML = '<option value="">No unknown face visible</option>';
    registerBtn.disabled = true;
    return;
  }
  for (const u of unknowns) {
    const opt = document.createElement('option');
    opt.value = u.id;
    opt.textContent = `Unknown ${u.id} - distance ${Number(u.distance).toFixed(3)}`;
    unknownSelect.appendChild(opt);
  }
  if ([...unknownSelect.options].some(o => o.value === selected)) unknownSelect.value = selected;
  registerBtn.disabled = false;
}

function renderFaces(faces) {
  facesList.innerHTML = '';
  if (!faces || faces.length === 0) {
    facesList.textContent = 'No face visible';
    faceSummary.textContent = 'No Face';
    faceDistance.textContent = 'Distance: -';
    return;
  }
  faceSummary.textContent = `${faces.length} face(s) visible`;
  const first = faces[0];
  faceDistance.textContent = first.distance == null ? 'Distance: -' : `First face distance: ${Number(first.distance).toFixed(3)}`;
  for (const f of faces) {
    const div = document.createElement('div');
    div.className = 'face-item';
    div.innerHTML = `<strong>#${f.index}: ${f.name}</strong><br>Distance: ${Number(f.distance).toFixed(3)}${f.id ? `<br>Unknown ID: ${f.id}` : ''}`;
    facesList.appendChild(div);
  }
}

function renderHands(hands) {
  handsList.innerHTML = '';
  if (!hands || hands.length === 0) {
    handsList.textContent = 'No hand visible';
    return;
  }
  for (const h of hands) {
    const div = document.createElement('div');
    div.className = 'hand-item';
    const fingers = (h.finger_details || []).map(f => `<span class="finger-pill ${f.open ? 'open' : ''}">${f.label}: ${f.open ? 'Open' : 'Closed'}</span>`).join('');
    div.innerHTML = `<strong>${h.hand}</strong><br>Fingers: ${h.fingers}<br>Gesture: ${h.gesture}<br>Pinch: ${h.pinch.active ? 'Active' : 'Inactive'} (${Number(h.pinch.ratio).toFixed(2)})<div class="finger-row">${fingers}</div>`;
    handsList.appendChild(div);
  }
}

function renderGestures(gestures) {
  gesturesList.innerHTML = '';
  if (!gestures || gestures.length === 0) {
    gesturesList.textContent = 'No custom gestures yet';
    return;
  }
  for (const g of gestures) {
    const div = document.createElement('div');
    div.className = 'gesture-item';
    const states = Object.entries(g.states || {}).map(([k,v]) => `${k}: ${v ? 'open' : 'closed'}`).join(', ');
    const btn = document.createElement('button');
    btn.className = 'danger';
    btn.textContent = 'Delete';
    btn.onclick = () => deleteGesture(g.name);
    div.innerHTML = `<strong>${g.name}</strong><br><span class="muted">${states}</span>`;
    div.appendChild(btn);
    gesturesList.appendChild(div);
  }
}

function pointerToScreen(pointer) {
  const rect = videoWrap.getBoundingClientRect();
  return { x: pointer.x * rect.width, y: pointer.y * rect.height };
}

function drawLine(from, to) {
  ctx.strokeStyle = drawColor ? drawColor.value : '#ffffff';
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

function addWord(text) {
  const rect = videoWrap.getBoundingClientRect();
  const id = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  const item = { id, text, x: rect.width * 0.42, y: rect.height * 0.42 };
  words.push(item);
  const node = document.createElement('div');
  node.className = 'word-token';
  node.dataset.id = id;
  node.textContent = text;
  wordLayer.appendChild(node);
  renderWords();
}

function renderWords() {
  for (const w of words) {
    const node = wordLayer.querySelector(`[data-id="${w.id}"]`);
    if (!node) continue;
    node.style.left = `${w.x}px`;
    node.style.top = `${w.y}px`;
    node.classList.toggle('grabbed', grabbedWordId === w.id);
  }
}

function findNearestWord(point) {
  let best = null;
  let bestDist = 90;
  for (const w of words) {
    const node = wordLayer.querySelector(`[data-id="${w.id}"]`);
    const width = node ? node.offsetWidth : 80;
    const height = node ? node.offsetHeight : 36;
    const cx = w.x + width / 2;
    const cy = w.y + height / 2;
    const dist = Math.hypot(point.x - cx, point.y - cy);
    if (dist < bestDist) { bestDist = dist; best = w; }
  }
  return best;
}

function handleHandInteraction(hands) {
  const hand = hands && hands[0];
  if (!hand || !hand.pointer) {
    cursorDot.style.display = 'none';
    lastDrawPoint = null;
    grabbedWordId = null;
    renderWords();
    return;
  }
  const point = pointerToScreen(hand.pointer);
  const pinch = !!hand.pinch.active;
  cursorDot.style.display = 'block';
  cursorDot.style.left = point.x + 'px';
  cursorDot.style.top = point.y + 'px';
  cursorDot.classList.toggle('pinch', pinch);

  if (mode === 'draw' || mode === 'write') {
    if (pinch) {
      if (lastDrawPoint) drawLine(lastDrawPoint, point);
      lastDrawPoint = point;
    } else {
      lastDrawPoint = null;
    }
    grabbedWordId = null;
  }

  if (mode === 'move') {
    lastDrawPoint = null;
    if (pinch && !lastPinch) {
      const nearest = findNearestWord(point);
      grabbedWordId = nearest ? nearest.id : null;
    }
    if (pinch && grabbedWordId) {
      const w = words.find(item => item.id === grabbedWordId);
      if (w) { w.x = point.x - 30; w.y = point.y - 18; }
    }
    if (!pinch) grabbedWordId = null;
    renderWords();
  }
  lastPinch = pinch;
}

async function postJson(url, payload) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  return res.json();
}

async function refreshState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    currentState = data;
    connectionStatus.textContent = 'Connected';
    renderFaces(data.faces);
    renderUnknowns(data.unknown_faces);
    renderHands(data.hands);
    renderPeople(data.people);
    renderGestures(data.custom_gestures);
    if (performanceText && data.performance) {
      performanceText.textContent = `Async recognition | Queue: ${data.performance.recognition_queue} | Threshold: ${data.performance.face_threshold}`;
    }
    handleHandInteraction(data.hands);
  } catch (e) {
    connectionStatus.textContent = 'Disconnected';
  }
}

registerBtn.addEventListener('click', async () => {
  setMessage(registerMessage, 'Registering...');
  const data = await postJson('/api/register_face', { name: personName.value, unknown_id: selectedUnknownId() });
  setMessage(registerMessage, data.message);
  if (data.ok) { personName.value = ''; renderPeople(data.people); }
});

el('renameBtn').addEventListener('click', async () => {
  const data = await postJson('/api/rename_person', { old_name: selectedPersonName(), new_name: renameInput.value });
  setMessage(peopleMessage, data.message);
  if (data.ok) { renameInput.value = ''; renderPeople(data.people); }
});

el('deletePersonBtn').addEventListener('click', async () => {
  if (!selectedPersonName()) return;
  const data = await postJson('/api/delete_person', { name: selectedPersonName() });
  setMessage(peopleMessage, data.message);
  renderPeople(data.people);
});

el('clearSamplesBtn').addEventListener('click', async () => {
  if (!selectedPersonName()) return;
  const data = await postJson('/api/clear_samples', { name: selectedPersonName() });
  setMessage(peopleMessage, data.message);
  renderPeople(data.people);
});

el('addSampleBtn').addEventListener('click', async () => {
  const data = await postJson('/api/add_face_sample', { name: selectedPersonName(), unknown_id: selectedUnknownId() });
  setMessage(peopleMessage, data.message);
  if (data.people) renderPeople(data.people);
});

el('addWordBtn').addEventListener('click', () => {
  const text = wordInput.value.trim();
  if (!text) return;
  addWord(text);
  wordInput.value = '';
});

el('clearCanvasBtn').addEventListener('click', () => {
  ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
});

el('clearWordsBtn').addEventListener('click', () => {
  words = [];
  grabbedWordId = null;
  wordLayer.innerHTML = '';
});

document.querySelectorAll('.tool').forEach(btn => {
  btn.addEventListener('click', () => {
    mode = btn.dataset.mode;
    document.querySelectorAll('.tool').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

el('saveGestureBtn').addEventListener('click', async () => {
  const states = {};
  document.querySelectorAll('[data-finger]').forEach(cb => { states[cb.dataset.finger] = cb.checked; });
  const data = await postJson('/api/custom_gestures', { name: el('gestureName').value, states });
  setMessage(gestureMessage, data.message);
  if (data.ok) { el('gestureName').value = ''; renderGestures(data.gestures); }
});

async function deleteGesture(name) {
  const res = await fetch('/api/custom_gestures', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
  const data = await res.json();
  setMessage(gestureMessage, data.message);
  renderGestures(data.gestures);
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();
refreshState();
setInterval(refreshState, 200);
