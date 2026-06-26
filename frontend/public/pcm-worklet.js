// pcm-worklet.js
//
// AudioWorklet processor for HealthVoice.
//
// Runs in a SEPARATE global scope (AudioWorkletGlobalScope). It cannot import
// anything and has no access to the DOM. It must be fully self-contained.
//
// The browser calls process() with 128-sample Float32 frames. We accumulate
// those into ~1024-sample batches and post each completed batch (a Float32Array)
// back to the main thread, where it is converted to Int16 PCM and shipped over
// the WebSocket.

const BATCH_SIZE = 1024;

class PCMWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(BATCH_SIZE);
    this._offset = 0;
  }

  process(inputs) {
    const input = inputs[0];

    // No input connected (e.g. just after start/stop) — keep the node alive.
    if (!input || input.length === 0) {
      return true;
    }

    // Mono: take the first channel. getUserMedia is requested with
    // channelCount: 1, so channel 0 is what we want.
    const channel = input[0];
    if (!channel) {
      return true;
    }

    for (let i = 0; i < channel.length; i++) {
      this._buffer[this._offset++] = channel[i];

      if (this._offset === BATCH_SIZE) {
        // Post a copy so the main thread owns its own buffer and we can
        // safely keep filling _buffer for the next batch.
        const batch = this._buffer.slice(0, BATCH_SIZE);
        this.port.postMessage(batch, [batch.buffer]);
        this._offset = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
