import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Html } from "@react-three/drei";
import * as THREE from "three";
import { viridis } from "../lib/colormap.js";

const STEP_CHUNK = 32;       // step-axis capacity grows in chunks -> stable scale
const SPAN = 60;             // world units the grid spans on each axis
const MAX_H = 20;            // tallest bar in world units
const GROW = 0.18;           // lerp factor for new-bar growth animation

const scaleVal = (v, log) => (log ? Math.log10(1 + Math.abs(v || 0)) : Math.abs(v || 0));

function Bars({ layers, points, logScale, onSelect, onRange, highlightSteps, resetKey, revealStep }) {
  const meshRef = useRef();
  const [hover, setHover] = useState(-1);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  const depthIndex = useMemo(
    () => new Map(layers.map((l, i) => [l.depth_idx, i])), [layers]);
  const nX = layers.length;

  // distinct step count + index map, single pass (no spread)
  const { stepIndex, nSteps } = useMemo(() => {
    const set = new Set();
    for (const p of points) set.add(p.step);
    const sorted = Array.from(set).sort((a, b) => a - b);
    return { stepIndex: new Map(sorted.map((s, i) => [s, i])), nSteps: sorted.length };
  }, [points]);

  // step-axis capacity in chunks so geometry scale only changes occasionally
  const capSteps = Math.max(STEP_CHUNK, Math.ceil(nSteps / STEP_CHUNK) * STEP_CHUNK);
  const capacity = Math.max(1, nX * capSteps);

  // persistent per-instance state across renders/frames
  const st = useRef({
    resetKey: null, capacity: 0, written: 0, prevCapSteps: 0,
    t: null, step: null, xi: null, si: null,
    targetH: null, curH: null, ids: null, positions: null,
    pending: new Set(),
    vmax: 1e-9,
  });

  const sx = SPAN / Math.max(1, nX);
  const sy = SPAN / Math.max(1, capSteps);

  const writeMatrix = (mesh, i, xi, si, h) => {
    const px = xi * sx - SPAN / 2, pz = si * sy - SPAN / 2;
    dummy.position.set(px, h / 2, pz);
    dummy.scale.set(sx * 0.82, Math.max(0.0001, h), sy * 0.82);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    return [px, h + 1, pz];
  };

  const writeColor = (mesh, i, t, step, hlSet) => {
    const [r, g, b] = viridis(t);
    tmpColor.setRGB(r, g, b);
    if (hlSet && !hlSet.has(step)) tmpColor.multiplyScalar(0.22);
    mesh.setColorAt(i, tmpColor);
  };

  const writeColorDim = (mesh, i, t, dim) => {
    const [r, g, b] = viridis(t);
    tmpColor.setRGB(r, g, b);
    if (dim !== 1) tmpColor.multiplyScalar(dim);
    mesh.setColorAt(i, tmpColor);
  };

  // (re)allocate persistent arrays on reset or capacity bump
  const ensureArrays = () => {
    const s = st.current;
    const reset = s.resetKey !== resetKey;
    const grew = s.capacity !== capacity;
    if (reset || grew || !s.t) {
      const old = reset ? null : s;
      const nt = new Float32Array(capacity), nstep = new Int32Array(capacity);
      const nxi = new Int32Array(capacity), nsi = new Int32Array(capacity);
      const ntH = new Float32Array(capacity), ncH = new Float32Array(capacity);
      if (old && old.t) {      // capacity grew: preserve already-written instances
        const n = Math.min(old.written, capacity);
        nt.set(old.t.subarray(0, n)); nstep.set(old.step.subarray(0, n));
        nxi.set(old.xi.subarray(0, n)); nsi.set(old.si.subarray(0, n));
        ntH.set(old.targetH.subarray(0, n)); ncH.set(old.curH.subarray(0, n));
      }
      s.t = nt; s.step = nstep; s.xi = nxi; s.si = nsi; s.targetH = ntH; s.curH = ncH;
      s.ids = old && old.ids ? old.ids.slice(0, capacity) : new Array(capacity);
      s.positions = old && old.positions ? old.positions.slice(0, capacity) : new Array(capacity);
      if (reset) { s.written = 0; s.vmax = 1e-9; s.pending = new Set(); }
      s.capacity = capacity;
      s.resetKey = resetKey;
    }
    return s;
  };

  // main write: append new instances, animate them up; reposition all if scale changed
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const s = ensureArrays();
    const hlSet = highlightSteps ? new Set(highlightSteps) : null;

    // running max over the (new) scaled values, single pass
    let batchMax = s.vmax;
    for (let i = s.written; i < points.length; i++) {
      const v = scaleVal(points[i].value, logScale);
      if (v > batchMax) batchMax = v;
    }
    const rescaleAll = batchMax !== s.vmax || s.prevCapSteps !== capSteps;
    s.vmax = batchMax;
    s.prevCapSteps = capSteps;
    onRange && onRange({ vmin: 0, vmax: s.vmax, logScale });

    const denom = Math.max(1e-12, s.vmax);
    const start = rescaleAll ? 0 : s.written;

    for (let i = start; i < points.length; i++) {
      const p = points[i];
      const xi = depthIndex.get(p.depth_idx);
      const si = stepIndex.get(p.step);
      if (xi == null || si == null) continue;
      const t = scaleVal(p.value, logScale) / denom;
      const h = 0.2 + t * MAX_H;
      s.t[i] = t; s.step[i] = p.step; s.xi[i] = xi; s.si[i] = si; s.targetH[i] = h;
      s.ids[i] = { layer: layers[xi].layer_name, step: p.step, value: p.value, kind: layers[xi].kind };
      const isNew = i >= s.written;
      if (isNew) { s.curH[i] = 0; s.pending.add(i); }       // grow from 0
      const drawH = rescaleAll && !isNew ? s.targetH[i] : s.curH[i];
      s.positions[i] = writeMatrix(mesh, i, xi, si, drawH);
      writeColor(mesh, i, t, p.step, hlSet);
    }
    s.written = points.length;
    mesh.count = points.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [points, logScale, resetKey, capacity, capSteps]);

  // style pass on highlight / playback-reveal change.
  // revealStep != null  -> cumulative reveal: only steps <= revealStep are shown
  //                        (current step bright, earlier steps dimmer) so playback
  //                        builds the surface up from empty.
  // revealStep == null  -> full surface, with optional highlight dimming.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const s = st.current;
    if (!s.t) return;
    const reveal = revealStep;
    if (reveal != null) s.pending.clear();   // playback owns heights; stop growth lerp
    const hlSet = highlightSteps ? new Set(highlightSteps) : null;
    for (let i = 0; i < s.written; i++) {
      if (reveal != null) {
        const visible = s.step[i] <= reveal;
        s.positions[i] = writeMatrix(mesh, i, s.xi[i], s.si[i], visible ? s.targetH[i] : 0.0001);
        const dim = !visible ? 0 : (s.step[i] === reveal ? 1 : 0.5);
        writeColorDim(mesh, i, s.t[i], dim);
      } else {
        s.positions[i] = writeMatrix(mesh, i, s.xi[i], s.si[i], s.targetH[i]);
        writeColorDim(mesh, i, s.t[i], hlSet && !hlSet.has(s.step[i]) ? 0.22 : 1);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [highlightSteps, revealStep]);

  // animate pending (newly added) bars growing to target height
  useFrame(() => {
    const mesh = meshRef.current;
    const s = st.current;
    if (!mesh || s.pending.size === 0 || !s.curH) return;
    const done = [];
    for (const i of s.pending) {
      const target = s.targetH[i];
      s.curH[i] += (target - s.curH[i]) * GROW;
      if (Math.abs(target - s.curH[i]) < 0.01) { s.curH[i] = target; done.push(i); }
      s.positions[i] = writeMatrix(mesh, i, s.xi[i], s.si[i], s.curH[i]);
    }
    for (const i of done) s.pending.delete(i);
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, capacity]}
        onPointerMove={(e) => { e.stopPropagation(); setHover(e.instanceId ?? -1); }}
        onPointerOut={() => setHover(-1)}
        onClick={(e) => { e.stopPropagation(); const id = st.current.ids?.[e.instanceId]; if (id) onSelect(id); }}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial toneMapped={false} roughness={0.5} metalness={0.05} />
      </instancedMesh>
      {hover >= 0 && st.current.ids?.[hover] && st.current.positions?.[hover] && (
        <Html position={st.current.positions[hover]} center distanceFactor={60} style={{ pointerEvents: "none" }}>
          <div style={tip}>
            <div style={{ fontWeight: 600 }}>{st.current.ids[hover].layer}</div>
            <div style={{ opacity: 0.8 }}>{st.current.ids[hover].kind} · step {st.current.ids[hover].step}</div>
            <div>{Number(st.current.ids[hover].value).toExponential(3)}</div>
          </div>
        </Html>
      )}
    </>
  );
}

const tip = {
  background: "rgba(10,14,22,0.92)", border: "1px solid #2a3344", borderRadius: 6,
  padding: "6px 8px", fontSize: 11, whiteSpace: "nowrap", color: "#e6e6e6",
  transform: "translateY(-8px)",
};

export default function Surface3D({ layers, points, onSelect, metric, logScale,
                                   onRange, highlightSteps, resetKey, revealStep,
                                   stepLabel = "step" }) {
  if (!points?.length || !layers?.length)
    return <div style={{ padding: 24, opacity: 0.6 }}>No trace data for this run yet.</div>;

  return (
    <Canvas camera={{ position: [55, 45, 55], fov: 50 }} style={{ height: "100%" }}>
      <color attach="background" args={["#0b0e14"]} />
      <ambientLight intensity={1.15} />
      <directionalLight position={[40, 60, 20]} intensity={0.7} />
      <directionalLight position={[-30, 40, -20]} intensity={0.4} />
      <Bars layers={layers} points={points} onSelect={onSelect} logScale={logScale}
            onRange={onRange} highlightSteps={highlightSteps} resetKey={resetKey}
            revealStep={revealStep} />
      <gridHelper args={[64, 16, "#2a3344", "#1a2230"]} />
      <Text position={[0, -2, 36]} fontSize={3} color="#7fd1ff">{stepLabel} →</Text>
      <Text position={[-36, -2, 0]} rotation={[0, Math.PI / 2, 0]} fontSize={3} color="#ffb37f">
        layer depth →
      </Text>
      <Text position={[-36, 12, -34]} fontSize={2.5} color="#9fe0a0">{metric} ↑</Text>
      <OrbitControls enableDamping />
    </Canvas>
  );
}
