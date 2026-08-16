/**
 * TTS Web Worker (ES Module) — runs OmniVoice inference via ONNX Runtime Web.
 * Uses @huggingface/transformers for proper Qwen2 BPE tokenization.
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.mjs';
import { AutoTokenizer, env as tfEnv } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';
import { estimateTargetTokens } from '../duration-estimator.js';
import { GpuPostProcessor } from './gpu-postprocess.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

// Maximize performance — multi-threading requires cross-origin isolation (COOP/COEP headers)
ort.env.wasm.numThreads = self.crossOriginIsolated ? (navigator.hardwareConcurrency || 4) : 1;
ort.env.wasm.simd = true;

// Configure transformers.js to load tokenizer from our server
tfEnv.allowLocalModels = false;

let mainSession = null;
let decoderSession = null;
let encoderSession = null;
let tokenizer = null;
let config = null;
let gpuPostProc = null;

// ─── Cancellation ───────────────────────────────────────────────────────────
// Job-scoped: each synthesize message snapshots the cancel counter when it
// ARRIVES (not when it starts), so a cancel posted while the job sits behind
// a queued encode still applies to it.

let cancelCounter = 0;
let activeCancelBaseline = 0;
let activeJobId = null;

function isCancelRequested() { return cancelCounter > activeCancelBaseline; }

class CancelledError extends Error {}

// Force a macrotask turn so an incoming 'cancel' message can be delivered
// between diffusion steps even on the WASM backend (whose session.run promise
// may resolve without yielding to the event loop).
const _yieldChannel = new MessageChannel();
let _yieldResolve = null;
_yieldChannel.port1.onmessage = () => { const r = _yieldResolve; _yieldResolve = null; if (r) r(); };
function yieldMacrotask() {
  return new Promise(res => { _yieldResolve = res; _yieldChannel.port2.postMessage(0); });
}

// ─── Cache API ─────────────────────────────────────────────────────────────

const CACHE_NAME = 'omnivoice-models-v1';

// ─── Fetch with progress + Cache API caching ──────────────────────────────

async function fetchWithProgress(url, onProgress, onCached) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    if (onCached) onCached(buf.byteLength);
    return buf;
  }

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Fetch failed: ${resp.status} for ${url}`);
  const contentLength = parseInt(resp.headers.get('Content-Length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (onProgress) onProgress(loaded, contentLength || null);
  }
  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.byteLength; }
  const buf = result.buffer;

  // Store in Cache API — no structured clone needed, stores as a Response blob
  try {
    await cache.put(url, new Response(buf, {
      headers: { 'Content-Length': String(buf.byteLength), 'Content-Type': 'application/octet-stream' }
    }));
  } catch (e) { console.warn('Cache store failed:', e); }
  return buf;
}

// ─── Tensor helper ──────────────────────────────────────────────────────────

function T(type, data, dims) { return new ort.Tensor(type, data, dims); }

// ─── Time steps (port of _get_time_steps) ───────────────────────────────────

function getTimeSteps(tStart, tEnd, numStep, tShift) {
  const steps = [];
  for (let i = 0; i <= numStep; i++) {
    let t = tStart + (tEnd - tStart) * (i / numStep);
    t = tShift * t / (1 + (tShift - 1) * t);
    steps.push(t);
  }
  return steps;
}

// ─── Log-softmax over a slice of a Float32Array ─────────────────────────────

// ─── Seeded PRNG (mulberry32) for deterministic generation ──────────────────

function mulberry32(seed) {
  let s = seed | 0;
  return function() {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let rng = Math.random; // default, overridden per generation

// Pre-allocated buffers for hot-path computation (avoids GC pressure)
const _cLP = new Float32Array(1025);
const _uLP = new Float32Array(1025);
const _g = new Float32Array(1025);

function logSoftmaxInto(arr, offset, len, out) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) { const v = arr[offset + i]; if (v > max) max = v; }
  let sum = 0;
  for (let i = 0; i < len; i++) sum += Math.exp(arr[offset + i] - max);
  const lse = max + Math.log(sum);
  for (let i = 0; i < len; i++) out[i] = arr[offset + i] - lse;
}

function cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, pred, scores) {
  const gScale1 = 1 + guidanceScale;
  for (let c = 0; c < C; c++) {
    const layerScore = layerPenalty * c;
    for (let t = 0; t < numTargetTokens; t++) {
      const cOff = (c * maxLen + targetOff + t) * V;
      const uOff = ((C + c) * maxLen + t) * V;
      logSoftmaxInto(logits, cOff, V, _cLP);
      logSoftmaxInto(logits, uOff, V, _uLP);
      let mx = -Infinity;
      for (let v = 0; v < V; v++) {
        const gv = gScale1 * _cLP[v] - guidanceScale * _uLP[v];
        _g[v] = gv;
        if (gv > mx) mx = gv;
      }
      let sm = 0;
      for (let v = 0; v < V; v++) sm += Math.exp(_g[v] - mx);
      const lse = mx + Math.log(sm);
      let bestV = 0, bestS = -Infinity;
      for (let v = 0; v < V; v++) {
        if (v === maskId) continue;
        const lp = _g[v] - lse;
        if (lp > bestS) { bestS = lp; bestV = v; }
      }
      const idx = c * numTargetTokens + t;
      pred[idx] = bestV;
      scores[idx] = bestS - layerScore;
    }
  }
}

// ─── Prepare inference inputs ───────────────────────────────────────────────

async function prepareInferenceInputs(text, numTargetTokens, tok, cfg, opts = {}) {
  const { refText = null, refAudioTokens = null, lang = null, instruct = null, denoise = true } = opts;
  const C = cfg.num_audio_codebook;
  const maskId = cfg.audio_mask_id;

  // Build style string
  let styleText = '';
  if (denoise) styleText += '<|denoise|>';
  styleText += `<|lang_start|>${lang || 'None'}<|lang_end|>`;
  styleText += `<|instruct_start|>${instruct || 'None'}<|instruct_end|>`;

  // Build text string
  let fullText = refText ? refText.trim() + ' ' + text.trim() : text.trim();
  fullText = fullText.replace(/[\r\n]+/g, '').replace(/[ \t]+/g, ' ');
  const wrappedText = `<|text_start|>${fullText}<|text_end|>`;

  // Tokenize using transformers.js (proper Qwen2 BPE)
  const styleEncoded = await tok(styleText, { add_special_tokens: false });
  const textEncoded = await tok(wrappedText, { add_special_tokens: false });
  // transformers.js returns Tensors — extract as plain number arrays
  const styleIds = Array.from(styleEncoded.input_ids.data, Number);
  const textIds = Array.from(textEncoded.input_ids.data, Number);

  // Sequence layout: [style | text | ref_audio? | target_masked]
  const refLen = refAudioTokens ? refAudioTokens[0].length : 0;
  const totalLen = styleIds.length + textIds.length + refLen + numTargetTokens;

  const inputIds = new BigInt64Array(C * totalLen);

  // Style tokens (replicated across codebooks)
  for (let c = 0; c < C; c++)
    for (let i = 0; i < styleIds.length; i++)
      inputIds[c * totalLen + i] = BigInt(styleIds[i]);

  // Text tokens
  const textOff = styleIds.length;
  for (let c = 0; c < C; c++)
    for (let i = 0; i < textIds.length; i++)
      inputIds[c * totalLen + textOff + i] = BigInt(textIds[i]);

  // Reference audio tokens
  const refOff = textOff + textIds.length;
  if (refAudioTokens) {
    for (let c = 0; c < C; c++)
      for (let t = 0; t < refLen; t++)
        inputIds[c * totalLen + refOff + t] = BigInt(refAudioTokens[c][t]);
  }

  // Target = all mask
  const targetOff = refOff + refLen;
  for (let c = 0; c < C; c++)
    for (let t = 0; t < numTargetTokens; t++)
      inputIds[c * totalLen + targetOff + t] = BigInt(maskId);

  // Audio mask: true for audio positions (ref + target)
  const audioMask = new Uint8Array(totalLen);
  const audioStart = refAudioTokens ? refOff : targetOff;
  for (let i = audioStart; i < totalLen; i++) audioMask[i] = 1;

  return { inputIds, audioMask, totalLen, numTargetTokens, targetOff, C };
}

// ─── Top-k unmask using partial selection ───────────────────────────────────

function topKUnmask(scores, pred, tokens, n, k) {
  // Find k-th largest score using nth_element-style partition
  // For small k (typically 2-300), a simple selection is fast enough
  const indices = new Int32Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (scores[i] > -Infinity) indices[count++] = i;
  }
  // Partial sort: only find top k
  for (let i = 0; i < Math.min(k, count); i++) {
    let maxIdx = i;
    for (let j = i + 1; j < count; j++) {
      if (scores[indices[j]] > scores[indices[maxIdx]]) maxIdx = j;
    }
    if (maxIdx !== i) { const tmp = indices[i]; indices[i] = indices[maxIdx]; indices[maxIdx] = tmp; }
    tokens[indices[i]] = BigInt(pred[indices[i]]);
  }
}

let pred_buf = null, scores_buf = null;

// ─── Iterative unmasking generation loop ────────────────────────────────────

async function generateIterative(inp, cfg, numStep, guidanceScale, tShift, layerPenalty = 5.0, posTemp = 5.0) {
  const { inputIds, audioMask, totalLen, numTargetTokens, targetOff, C } = inp;
  const maskId = cfg.audio_mask_id;
  const V = cfg.audio_vocab_size;

  const condLen = totalLen;
  const uncondLen = numTargetTokens;
  const maxLen = condLen;

  // Batch input_ids: (2, C, maxLen) — cond + uncond
  const bIds = new BigInt64Array(2 * C * maxLen).fill(BigInt(maskId));
  for (let c = 0; c < C; c++)
    for (let s = 0; s < condLen; s++)
      bIds[c * maxLen + s] = inputIds[c * totalLen + s];
  for (let c = 0; c < C; c++)
    for (let t = 0; t < uncondLen; t++)
      bIds[(C + c) * maxLen + t] = inputIds[c * totalLen + targetOff + t];

  // Batch audio_mask: (2, maxLen)
  const bMask = new Uint8Array(2 * maxLen);
  for (let s = 0; s < condLen; s++) bMask[s] = audioMask[s];
  for (let t = 0; t < uncondLen; t++) bMask[maxLen + t] = 1;

  // Batch attention_mask: (2, 1, maxLen, maxLen)
  const bAttn = new Uint8Array(2 * maxLen * maxLen);
  for (let q = 0; q < condLen; q++)
    for (let k = 0; k < condLen; k++)
      bAttn[q * maxLen + k] = 1;
  for (let q = 0; q < uncondLen; q++)
    for (let k = 0; k < uncondLen; k++)
      bAttn[maxLen * maxLen + q * maxLen + k] = 1;
  for (let p = uncondLen; p < maxLen; p++)
    bAttn[maxLen * maxLen + p * maxLen + p] = 1;

  // Position IDs required by the exported Qwen3 ONNX model
const bPos = new BigInt64Array(2 * maxLen);

for (let b = 0; b < 2; b++) {
  for (let s = 0; s < maxLen; s++) {
    bPos[b * maxLen + s] = BigInt(s);
  }
}

  // Token state
  const tokens = new BigInt64Array(C * numTargetTokens).fill(BigInt(maskId));
  pred_buf = null; scores_buf = null;

  // Unmasking schedule
  // Python passes num_step+1 to _get_time_steps which creates linspace(0,1,num_step+2)
  const timesteps = getTimeSteps(0, 1, numStep + 1, tShift);
  const totalMask = numTargetTokens * C;
  let rem = totalMask;
  const sched = [];
  for (let s = 0; s < numStep; s++) {
    const n = s === numStep - 1 ? rem : Math.min(Math.ceil(totalMask * (timesteps[s + 1] - timesteps[s])), rem);
    sched.push(n); rem -= n;
  }

  if (gpuPostProc) {
    try { gpuPostProc.prepare(C, maxLen, V, numTargetTokens); }
    catch (e) { console.warn('[gpu-postprocess] prepare failed:', e.message); gpuPostProc.destroy(); gpuPostProc = null; }
  }

  let totalInferenceMs = 0, totalModelMs = 0, totalGpuPPMs = 0;
  for (let step = 0; step < numStep; step++) {
    // Let a pending 'cancel' message land, then honor it between steps
    await yieldMacrotask();
    if (isCancelRequested()) throw new CancelledError('cancelled');
    const k = sched[step];
    if (k <= 0) continue;
    const stepT0 = performance.now();

    const modelT0 = performance.now();
    const results = await mainSession.run({
      input_ids: T('int64', bIds, [2, C, maxLen]),
      audio_mask: T('bool', bMask, [2, maxLen]),
      attention_mask: T('bool', bAttn, [2, 1, maxLen, maxLen]),
      position_ids: T('int64', bPos, [2, maxLen]),
    });
    const logits = results.logits.data; // (2, C, maxLen, V)
    totalModelMs += performance.now() - modelT0;

    const nPos = C * numTargetTokens;
    const pred = step === 0 ? new Int32Array(nPos) : pred_buf;
    const scores = step === 0 ? new Float32Array(nPos) : scores_buf;
    if (step === 0) { pred_buf = pred; scores_buf = scores; }

    const ppT0 = performance.now();
    if (gpuPostProc) {
      try {
        await gpuPostProc.run(logits, {
          C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty
        }, pred, scores);
        // On first step, benchmark CPU too and keep whichever is faster
        if (step === 0) {
          const gpuMs = performance.now() - ppT0;
          const cpuPred = new Int32Array(nPos);
          const cpuScores = new Float32Array(nPos);
          const cpuT0 = performance.now();
          cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, cpuPred, cpuScores);
          const cpuMs = performance.now() - cpuT0;
          if (cpuMs < gpuMs) {
            console.log(`[gpu-postprocess] CPU faster (${cpuMs.toFixed(0)}ms) than GPU (${gpuMs.toFixed(0)}ms), switching to CPU`);
            // Use CPU results for this step
            pred.set(cpuPred);
            scores.set(cpuScores);
            gpuPostProc.destroy();
            gpuPostProc = null;
          } else {
            console.log(`[gpu-postprocess] GPU (${gpuMs.toFixed(0)}ms) faster than CPU (${cpuMs.toFixed(0)}ms), keeping GPU`);
          }
        }
      } catch (e) {
        console.warn('[gpu-postprocess] dispatch failed, falling back to CPU:', e.message);
        gpuPostProc.destroy();
        gpuPostProc = null;
        cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, pred, scores);
      }
    } else {
      cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, pred, scores);
    }
    totalGpuPPMs += performance.now() - ppT0;

    // Gumbel noise + mask already-unmasked (fused)
    const bigMaskId = BigInt(maskId);
    if (posTemp > 0) {
      const invTemp = 1 / posTemp;
      for (let i = 0; i < nPos; i++) {
        if (tokens[i] !== bigMaskId) { scores[i] = -Infinity; continue; }
        scores[i] = scores[i] * invTemp + (-Math.log(-Math.log(rng() + 1e-10) + 1e-10));
      }
    } else {
      for (let i = 0; i < nPos; i++)
        if (tokens[i] !== bigMaskId) scores[i] = -Infinity;
    }

    // Partial top-k using quickselect instead of full sort
    topKUnmask(scores, pred, tokens, nPos, k);


    // Update batch inputs
    for (let c = 0; c < C; c++)
      for (let t = 0; t < numTargetTokens; t++) {
        const v = tokens[c * numTargetTokens + t];
        bIds[c * maxLen + targetOff + t] = v;
        bIds[(C + c) * maxLen + t] = v;
      }

    const stepMs = performance.now() - stepT0;
    totalInferenceMs += stepMs;
    postMessage({
      type: 'progress', stage: 'generating', jobId: activeJobId,
      step: step + 1, numStep, stepMs: Math.round(stepMs),
      detail: `Step ${step + 1}/${numStep} (${stepMs.toFixed(0)}ms)`,
    });
  }
  const jsMs = totalInferenceMs - totalModelMs;
  const ppLabel = gpuPostProc ? 'GPU-PP' : 'CPU-PP';
  console.log(`[perf] ${numStep} steps in ${totalInferenceMs.toFixed(0)}ms total | model: ${totalModelMs.toFixed(0)}ms (${(totalModelMs/numStep).toFixed(0)}ms/step) | ${ppLabel}: ${totalGpuPPMs.toFixed(0)}ms (${(totalGpuPPMs/numStep).toFixed(0)}ms/step) | JS-other: ${(jsMs - totalGpuPPMs).toFixed(0)}ms`);

  return tokens;
}

// ─── Decode & post-process ──────────────────────────────────────────────────

async function decodeTokens(tokens, C, T) {
  postMessage({ type: 'progress', stage: 'decoding', detail: 'Converting tokens to audio...' });
  const codes = new BigInt64Array(C * T);
  codes.set(tokens);
  const r = await decoderSession.run({ audio_codes: new ort.Tensor('int64', codes, [1, C, T]) });
  return r.audio_values.data;
}

function postProcessAudio(pcm, sr, normalize = true) {
  const thresh = 0.005, margin = Math.floor(sr * 0.02);
  let start = 0, end = pcm.length;
  for (let i = 0; i < pcm.length; i++) if (Math.abs(pcm[i]) > thresh) { start = Math.max(0, i - margin); break; }
  for (let i = pcm.length - 1; i >= 0; i--) if (Math.abs(pcm[i]) > thresh) { end = Math.min(pcm.length, i + margin); break; }
  const out = pcm.slice(start, end);
  let peak = 0;
  for (let i = 0; i < out.length; i++) { const a = Math.abs(out[i]); if (a > peak) peak = a; }
  if (normalize && peak > 1e-6) {
    const s = 0.5 / peak;
    for (let i = 0; i < out.length; i++) out[i] *= s;
    peak = 0.5;
  }
  return { pcm: out, peak };
}

// ─── Init ───────────────────────────────────────────────────────────────────


async function init(modelBaseUrl, forceCPU) {
  try {
    let hasWorkingGPU = false;

    if (
      !forceCPU &&
      typeof navigator !== 'undefined' &&
      navigator.gpu
    ) {
      try {
        const adapter =
          await navigator.gpu.requestAdapter();

        hasWorkingGPU = !!adapter;
      } catch {}
    }

    if (!hasWorkingGPU) {
      postMessage({
        type: 'progress',
        stage: 'loading',
        detail:
          'WebGPU غير متاح — سيتم استخدام CPU'
      });
    }

    postMessage({
      type: 'progress',
      stage: 'loading',
      phase: 'config',
      detail: 'Loading Egyptian VoiceTut config...'
    });

    const manifestResponse = await fetch(
      `${modelBaseUrl}/model-chunks.json`
    );

    if (!manifestResponse.ok) {
      throw new Error(
        `Manifest HTTP ${manifestResponse.status}`
      );
    }

    const manifest =
      await manifestResponse.json();

    const configResponse = await fetch(
      `${modelBaseUrl}/${manifest.config}`
    );

    if (!configResponse.ok) {
      throw new Error(
        `Config HTTP ${configResponse.status}`
      );
    }

    config = await configResponse.json();

    postMessage({
      type: 'progress',
      stage: 'loading',
      phase: 'tokenizer',
      detail: 'Loading text tokenizer...'
    });

    tokenizer =
      await AutoTokenizer.from_pretrained(
        'Gigsu/vocoloco-onnx'
      );

    const mainModelUrl =
      `${modelBaseUrl}/${manifest.mainModel}`;

    const decoderModelUrl =
      `${modelBaseUrl}/${manifest.decoderModel}`;

    const decoderDataUrl =
      `${modelBaseUrl}/${manifest.decoderData}`;

    const chunkUrls =
      manifest.mainChunks.map(
        x => `${modelBaseUrl}/${x.file}`
      );

    const urls = [
      mainModelUrl,
      ...chunkUrls,
      decoderModelUrl,
      decoderDataUrl
    ];

    const cache =
      await caches.open(CACHE_NAME);

    const checks =
      await Promise.all(
        urls.map(
          u => cache.match(u)
        )
      );

    let cachedBytes = 0;

    for (const hit of checks) {
      if (!hit) continue;

      const n = parseInt(
        hit.headers.get(
          'Content-Length'
        ) || '0',
        10
      );

      if (n > 0) {
        cachedBytes += n;
      }
    }

    const uncachedCount =
      checks.filter(
        x => !x
      ).length;

    postMessage({
      type: 'plan',
      firstRun:
        uncachedCount === urls.length,
      resuming:
        uncachedCount > 0 &&
        uncachedCount < urls.length,
      totalBytes:
        manifest.totalDownloadBytes,
      cachedBytes,
      fileCount: urls.length,
      filesToDownload:
        uncachedCount
    });

    let completedBytes =
      cachedBytes;

    async function loadOne(
      url,
      label,
      index
    ) {
      const cached =
        !!checks[index];

      let lastPost = 0;

      const buf =
        await fetchWithProgress(
          url,
          (loaded, total) => {
            const now =
              performance.now();

            if (
              now - lastPost < 150
            ) {
              return;
            }

            lastPost = now;

            postMessage({
              type: 'progress',
              stage: 'downloading',
              loadedBytes:
                completedBytes +
                loaded,
              totalBytes:
                manifest.totalDownloadBytes,
              fileIndex:
                index + 1,
              fileCount:
                urls.length,
              file:
                url.split('/').pop(),
              detail:
                `${label}: ` +
                `${Math.round(
                  loaded / 1e6
                )}/` +
                `${total
                  ? Math.round(
                      total / 1e6
                    )
                  : '?'} MB`
            });
          },
          null
        );

      if (!cached) {
        completedBytes +=
          buf.byteLength;
      }

      postMessage({
        type: 'progress',
        stage: 'downloading',
        loadedBytes:
          completedBytes,
        totalBytes:
          manifest.totalDownloadBytes,
        fileIndex:
          index + 1,
        fileCount:
          urls.length,
        file:
          url.split('/').pop(),
        detail:
          `${label} complete`
      });

      return buf;
    }

    let fileIndex = 0;

    const mainModelBuf =
      await loadOne(
        mainModelUrl,
        'VoiceTut model',
        fileIndex++
      );

    postMessage({
      type: 'progress',
      stage: 'loading',
      detail:
        'Assembling Egyptian model...'
    });

    const mainData =
      new Uint8Array(
        manifest.mainDataSize
      );

    let offset = 0;

    for (
      let i = 0;
      i < chunkUrls.length;
      i++
    ) {
      const buf =
        await loadOne(
          chunkUrls[i],
          `Model part ${i + 1}/` +
          `${chunkUrls.length}`,
          fileIndex++
        );

      const bytes =
        new Uint8Array(buf);

      mainData.set(
        bytes,
        offset
      );

      offset +=
        bytes.byteLength;
    }

    if (
      offset !==
      manifest.mainDataSize
    ) {
      throw new Error(
        'VoiceTut model size mismatch: ' +
        `${offset} != ` +
        `${manifest.mainDataSize}`
      );
    }

    const decoderModelBuf =
      await loadOne(
        decoderModelUrl,
        'Audio decoder',
        fileIndex++
      );

    const decoderDataBuf =
      await loadOne(
        decoderDataUrl,
        'Audio decoder data',
        fileIndex++
      );

    const mainExternalData = [{
      path:
        manifest.mainExternalPath,
      data:
        mainData.buffer
    }];

    postMessage({
      type: 'progress',
      stage: 'loading',
      phase: 'session-main',
      detail:
        'Starting Egyptian VoiceTut engine...'
    });

    let actualBackend = 'cpu';

    if (hasWorkingGPU) {
      try {
        mainSession =
          await ort.InferenceSession.create(
            mainModelBuf,
            {
              executionProviders:
                ['webgpu'],
              externalData:
                mainExternalData,
              graphOptimizationLevel:
                'all',
              enableCpuMemArena:
                true
            }
          );

        actualBackend =
          'webgpu';

      } catch (e) {
        console.warn(
          'VoiceTut WebGPU failed; ' +
          'falling back to WASM:',
          e.message
        );

        mainSession = null;
      }
    }

    if (!mainSession) {
      mainSession =
        await ort.InferenceSession.create(
          mainModelBuf,
          {
            executionProviders:
              ['wasm'],
            externalData:
              mainExternalData,
            graphOptimizationLevel:
              'all',
            enableCpuMemArena:
              true
          }
        );

      actualBackend = 'cpu';
    }

    if (
      actualBackend === 'cpu' &&
      hasWorkingGPU
    ) {
      try {
        gpuPostProc =
          new GpuPostProcessor();

        await gpuPostProc.init();
      } catch {
        gpuPostProc = null;
      }
    }

    postMessage({
      type: 'progress',
      stage: 'loading',
      phase: 'session-decoder',
      detail:
        'Starting audio decoder...'
    });

    const decoderOptions = {
      executionProviders:
        actualBackend === 'webgpu'
          ? ['webgpu', 'wasm']
          : ['wasm'],

      externalData: [{
        path:
          manifest.decoderExternalPath,
        data:
          decoderDataBuf
      }],

      graphOptimizationLevel:
        'all'
    };

    try {
      decoderSession =
        await ort.InferenceSession.create(
          decoderModelBuf,
          decoderOptions
        );

    } catch (e) {
      console.warn(
        'Decoder GPU failed; ' +
        'using WASM:',
        e.message
      );

      decoderSession =
        await ort.InferenceSession.create(
          decoderModelBuf,
          {
            executionProviders:
              ['wasm'],

            externalData: [{
              path:
                manifest.decoderExternalPath,
              data:
                decoderDataBuf
            }],

            graphOptimizationLevel:
              'all'
          }
        );
    }

    // Encoder intentionally omitted:
    // Mohamed voice tokens are already precomputed.
    encoderSession = null;

    postMessage({
      type: 'progress',
      stage: 'loading',
      phase: 'warmup',
      detail:
        'Warming up Egyptian voice...'
    });

    try {
      const dummyIds =
        new BigInt64Array(
          2 * 8 * 4
        ).fill(1024n);

      const dummyMask =
        new Uint8Array(
          2 * 4
        );

      const dummyAttn =
        new Uint8Array(
          2 * 1 * 4 * 4
        ).fill(1);

      await mainSession.run({
        input_ids:
          new ort.Tensor(
            'int64',
            dummyIds,
            [2, 8, 4]
          ),

        audio_mask:
          new ort.Tensor(
            'bool',
            dummyMask,
            [2, 4]
          ),

        attention_mask:
          new ort.Tensor(
            'bool',
            dummyAttn,
            [2, 1, 4, 4]
          )
      });

      const dummyCodes =
        new BigInt64Array(
          8 * 2
        ).fill(0n);

      await decoderSession.run({
        audio_codes:
          new ort.Tensor(
            'int64',
            dummyCodes,
            [1, 8, 2]
          )
      });

    } catch (e) {
      console.warn(
        'Warm-up warning:',
        e.message
      );
    }

    postMessage({
      type: 'ready',
      backend:
        actualBackend,
      encoderAvailable:
        false
    });

  } catch (err) {
    console.error(err);

    postMessage({
      type: 'error',
      message:
        `Init failed: ${err.message}`
    });
  }
}


// ─── Reference audio encoding ───────────────────────────────────────────────

// Runs the ~654 MB encoder over 24 kHz mono PCM and returns the audio tokens
// as a flat Int32Array [C*T] (codebook-major — same layout as the ONNX output).
async function encodeRefPcm(refAudio) {
  const pcmF32 = new Float32Array(refAudio);
  // Clip to hop_length alignment (hop=960 for 24kHz)
  const hopLength = 960;
  const clipLen = pcmF32.length - (pcmF32.length % hopLength);
  const aligned = pcmF32.slice(0, clipLen);
  const inputTensor = new ort.Tensor('float32', aligned, [1, 1, aligned.length]);
  const encResult = await encoderSession.run({ input_values: inputTensor });
  const codesData = encResult.audio_codes.data; // BigInt64Array
  const codeDims = encResult.audio_codes.dims; // [1, C, T]
  const C = Number(codeDims[1]);
  const tokenCount = Number(codeDims[2]);
  const tokens = new Int32Array(C * tokenCount);
  for (let i = 0; i < tokens.length; i++) tokens[i] = Number(codesData[i]);
  return { tokens, tokenCount, numCodebooks: C, duration: aligned.length / config.sampling_rate };
}

// Expand a flat Int32Array [C*T] into the nested [C][T] number arrays that
// prepareInferenceInputs consumes.
function expandTokens(flat, C) {
  const T = Math.floor(flat.length / C);
  const out = [];
  for (let c = 0; c < C; c++) {
    const row = new Array(T);
    for (let t = 0; t < T; t++) row[t] = flat[c * T + t];
    out.push(row);
  }
  return out;
}

async function encodeReference({ requestId, refAudio }) {
  if (!encoderSession) {
    postMessage({ type: 'encode-error', requestId, code: 'encoder-unavailable', message: 'Voice encoder not available on this device' });
    return;
  }
  try {
    postMessage({ type: 'progress', stage: 'encoding', detail: 'Analyzing voice...' });
    const r = await encodeRefPcm(refAudio);
    postMessage({ type: 'encoded', requestId, tokens: r.tokens, tokenCount: r.tokenCount, numCodebooks: r.numCodebooks, duration: r.duration }, [r.tokens.buffer]);
  } catch (err) {
    postMessage({ type: 'encode-error', requestId, code: 'encode-failed', message: err.message });
  }
}

// ─── Synthesize ─────────────────────────────────────────────────────────────

async function synthesize(params) {
  const {
    jobId = null,
    text, lang = null, refAudio = null, refText = null, refTokens = null,
    instruct = null,
    numStep = 20, guidanceScale = 4.0, tShift = 0.05, speed = 1.0,
    seed = null,
    returnTokens = false, normalize = true,
  } = params;

  try {
    // Use seeded PRNG for deterministic output when seed is provided
    rng = seed != null ? mulberry32(seed) : Math.random;
    const C = config.num_audio_codebook;

    // Resolve reference tokens for voice cloning:
    // 1. pre-encoded tokens (cached voice / chunk chaining) — no encoder needed
    // 2. raw PCM + encoder
    // 3. raw PCM without encoder — warn and proceed uncloned
    let refAudioTokens = null;
    if (refTokens && refTokens.length >= C) {
      refAudioTokens = expandTokens(refTokens, C);
      postMessage({ type: 'progress', stage: 'encoding', detail: `Using cached voice (${refAudioTokens[0].length} tokens)` });
    } else if (refAudio && encoderSession) {
      postMessage({ type: 'progress', stage: 'encoding', detail: 'Encoding reference audio...' });
      const enc = await encodeRefPcm(refAudio);
      refAudioTokens = expandTokens(enc.tokens, enc.numCodebooks);
      postMessage({ type: 'progress', stage: 'encoding', detail: `Encoded: ${enc.tokenCount} tokens (${enc.duration.toFixed(1)}s)` });
    } else if (refAudio && !encoderSession) {
      postMessage({ type: 'progress', stage: 'warning', detail: 'Voice cloning unavailable on this device (not enough memory for encoder)' });
    }

    // Duration estimation — the (refText, refTokens) pair must stay consistent:
    // mixing a real token count with the default text (or vice versa) skews the
    // estimate ~10x. estimateTargetTokens falls back to its internal defaults
    // whenever either half is missing.
    const estRefText = refAudioTokens ? refText : null;
    const estRefTokens = refAudioTokens ? refAudioTokens[0].length : null;
    let numTargetTokens = Math.min(estimateTargetTokens(text, estRefText, estRefTokens, speed), 700);

    postMessage({ type: 'progress', stage: 'preparing', detail: `Target: ${numTargetTokens} tokens` });

    const inputs = await prepareInferenceInputs(text, numTargetTokens, tokenizer, config, {
      lang, instruct, refText: refAudioTokens ? refText : null, refAudioTokens,
      denoise: true,
    });

    const tokens = await generateIterative(inputs, config, numStep, guidanceScale, tShift);

    if (isCancelRequested()) throw new CancelledError('cancelled');
    const rawPcm = await decodeTokens(tokens, C, numTargetTokens);

    postMessage({ type: 'progress', stage: 'postprocessing', detail: 'Processing audio...' });
    const { pcm, peak } = postProcessAudio(rawPcm, config.sampling_rate, normalize);

    const reply = { type: 'audio', jobId, pcm, sampleRate: config.sampling_rate, peak };
    const transfers = [pcm.buffer];
    if (returnTokens) {
      // Generated tokens are already valid reference-audio tokens — expose them
      // flat so the app can chain them as the voice reference for later chunks.
      const flat = new Int32Array(tokens.length);
      for (let i = 0; i < tokens.length; i++) flat[i] = Number(tokens[i]);
      reply.tokens = flat;
      reply.tokenCount = numTargetTokens;
      transfers.push(flat.buffer);
    }
    postMessage(reply, transfers);
  } catch (err) {
    if (err instanceof CancelledError) {
      postMessage({ type: 'cancelled', jobId });
      return;
    }
    postMessage({ type: 'error', jobId, message: `Synthesis failed: ${err.message}\n${err.stack}` });
  }
}

// ─── Message handler ────────────────────────────────────────────────────────
// 'cancel' is handled synchronously (never queued) so it can interrupt a
// running synthesis; everything else is serialized through a job queue since
// the pipeline shares mutable module-level buffers.

let jobQueue = Promise.resolve();

async function handleMessage(msg) {
  if (msg.type === 'init') {
    await init(msg.modelBaseUrl, msg.forceCPU);
  } else if (msg.type === 'synthesize') {
    activeJobId = msg.jobId ?? null;
    activeCancelBaseline = msg._cancelBaseline ?? cancelCounter;
    await synthesize(msg);
    activeJobId = null;
  } else if (msg.type === 'encode-reference') {
    await encodeReference(msg);
  }
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'cancel') { cancelCounter++; return; }
  // Snapshot the cancel counter at arrival so a cancel that lands while this
  // job is still queued (e.g. behind an encode) is honored when it runs.
  if (msg.type === 'synthesize') msg._cancelBaseline = cancelCounter;
  jobQueue = jobQueue.then(() => handleMessage(msg)).catch((err) => console.error('[worker] job failed:', err));
};
