import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";

const viewSelect = document.getElementById("view-select");
const frameSelect = document.getElementById("frame-select");
const pointSize = document.getElementById("point-size");
const fitButton = document.getElementById("fit-view");
const showMasked = document.getElementById("show-masked");
const viewport = document.getElementById("viewport");
const status = document.getElementById("status");
const canvas = document.getElementById("point-cloud");

let renderer;
let camera;
let controls;
let scene;
let cloud;
let views;
let heightMin;
let heightMax;

function fail(error) {
  status.hidden = false;
  status.dataset.state = "error";
  status.textContent = error.message || String(error);
  document.body.dataset.ready = "false";
}

function render() {
  renderer.render(scene, camera);
}

function resize() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  if (width === 0 || height === 0) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  render();
}

function fitView() {
  if (!cloud) return;
  const sphere = cloud.geometry.boundingSphere;
  const radius = Math.max(sphere.radius, 0.5);
  const verticalFov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
  const distance = radius / Math.sin(Math.min(verticalFov, horizontalFov) / 2) * 1.12;
  const direction = camera.position.clone().sub(controls.target).normalize();
  if (direction.lengthSq() === 0) direction.set(1, -1, 0.8).normalize();
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(direction, distance);
  camera.near = Math.max(distance / 10000, 0.01);
  camera.far = Math.max(distance * 100, 1000);
  camera.updateProjectionMatrix();
  controls.update();
  render();
}

function selectView({ fit = false } = {}) {
  const view = views[Number(viewSelect.value)];
  const positions = view[frameSelect.value];
  const originalPositions = view.origin_coord;
  if (!Array.isArray(positions) || positions.length === 0 || originalPositions.length !== positions.length) {
    throw new Error(`View "${view.name}" has no valid point coordinates.`);
  }
  const hasMask = Array.isArray(view.mask);
  if (hasMask && view.mask.length !== positions.length) {
    throw new Error(`View "${view.name}" has a mask that does not match its points.`);
  }
  showMasked.disabled = !hasMask;
  const coordinates = new Float32Array(positions.length * 3);
  const colors = new Float32Array(positions.length * 3);
  const color = new THREE.Color();
  const heightRange = Math.max(heightMax - heightMin, 1e-6);
  for (let i = 0; i < positions.length; i++) {
    coordinates.set(positions[i], i * 3);
    // A point keeps its original-height color across augmentations and crops.
    if (hasMask && showMasked.checked && view.mask[i]) {
      color.setHex(0xff4d9b);
    } else {
      const t = THREE.MathUtils.clamp((originalPositions[i][2] - heightMin) / heightRange, 0, 1);
      color.setHSL((1 - t) * 0.66, 0.85, 0.57);
    }
    color.toArray(colors, i * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(coordinates, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  if (cloud) {
    scene.remove(cloud);
    cloud.geometry.dispose();
    cloud.material.dispose();
  }
  cloud = new THREE.Points(geometry, new THREE.PointsMaterial({
    size: Number(pointSize.value),
    sizeAttenuation: false,
    vertexColors: true,
  }));
  scene.add(cloud);
  const countUnit = view.grid_size != null ? "occupied voxels (one point each)" : "points";
  document.getElementById("view-info").textContent = `${view.name} · ${positions.length.toLocaleString()} ${countUnit} · ${frameSelect.options[frameSelect.selectedIndex].text}`;
  const maskedCount = hasMask ? view.mask.filter(Boolean).length : 0;
  const scheduleInfo = view.mask_step != null && view.mask_total_steps != null
    ? ` · training step ${view.mask_step.toLocaleString()} / ${view.mask_total_steps.toLocaleString()}`
    : "";
  document.getElementById("mask-info").textContent = hasMask
    ? `${maskedCount.toLocaleString()} / ${positions.length.toLocaleString()} points selected for feature masking (${(100 * maskedCount / positions.length).toFixed(1)}%) · patch width ${view.mask_size} m · requested occupied-patch ratio ${(100 * view.mask_ratio).toFixed(1)}%${scheduleInfo}. All points retained. ${showMasked.checked ? "Pink marks" : "Highlight off; pink would mark"} features replaced by a learned token in the model.`
    : "";
  canvas.dataset.pointCount = String(positions.length);
  canvas.dataset.maskedCount = String(maskedCount);
  canvas.dataset.view = view.name;
  canvas.dataset.frame = frameSelect.value;
  if (fit) fitView();
  else render();
}

async function main() {
  const response = await fetch("/api/views");
  if (!response.ok) throw new Error(`Could not load views: HTTP ${response.status}. Check the terminal running the viewer.`);
  const payload = await response.json();
  views = payload.views;
  if (!Array.isArray(views) || views.length === 0) throw new Error("The server returned no point cloud views.");
  const original = views.find((view) => view.kind === "original") || views[0];
  const comparison = views.find((view) => view.kind === "shared") || original;
  heightMin = Infinity;
  heightMax = -Infinity;
  for (const position of original.origin_coord) {
    heightMin = Math.min(heightMin, position[2]);
    heightMax = Math.max(heightMax, position[2]);
  }
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  } catch {
    throw new Error("WebGL is unavailable. Enable browser hardware acceleration or use a browser with WebGL 2 support.");
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x10151c);
  scene = new THREE.Scene();
  scene.add(new THREE.AxesHelper(2));
  camera = new THREE.PerspectiveCamera(50, 1, 0.01, 10000);
  camera.up.set(0, 0, 1);
  camera.position.set(1, -1, 0.8);
  controls = new OrbitControls(camera, canvas);
  controls.addEventListener("change", render);
  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    fail(new Error("The browser lost its WebGL context. Reload this page to restart the viewer."));
  });

  views.forEach((view, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${view.name} (${view.count.toLocaleString()})`;
    viewSelect.appendChild(option);
  });
  viewSelect.value = String(views.indexOf(comparison));
  frameSelect.value = "origin_coord";
  viewSelect.addEventListener("change", () => {
    try { selectView(); } catch (error) { fail(error); }
  });
  frameSelect.addEventListener("change", () => {
    try { selectView({ fit: true }); } catch (error) { fail(error); }
  });
  pointSize.addEventListener("input", () => {
    document.getElementById("point-size-value").textContent = `${pointSize.value} px`;
    if (cloud) {
      cloud.material.size = Number(pointSize.value);
      render();
    }
  });
  showMasked.addEventListener("change", () => {
    try { selectView(); } catch (error) { fail(error); }
  });
  fitButton.addEventListener("click", fitView);
  const meta = payload.meta || {};
  const details = [meta.coordinate_frame, meta.version, meta.grid_size != null ? `voxel ${meta.grid_size} m` : null, meta.seed != null ? `seed ${meta.seed}` : null].filter(Boolean);
  document.getElementById("sample-info").textContent = `Sample: ${meta.sample_token || "unknown"}${details.length ? " · " + details.join(" · ") : ""}`;
  document.getElementById("color-info").textContent = `Color: original height, blue (${heightMin.toFixed(1)} m) → red (${heightMax.toFixed(1)} m), shared by every view. Axes: X red, Y green, Z blue.`;
  resize();
  selectView({ fit: true });
  new ResizeObserver(resize).observe(viewport);
  viewSelect.disabled = false;
  frameSelect.disabled = false;
  fitButton.disabled = false;
  status.hidden = true;
  status.dataset.state = "ready";
  document.body.dataset.ready = "true";
}

main().catch(fail);
