const app = getApp();

Page({
  data: {
    devicePosition: "front",
    flashMode: "off",
    connected: false,
    streaming: false,
    stats: {
      fps: 0,
      detectionCount: 0
    }
  },
  onReady() {
    console.log('[onReady] Page ready, initializing components...');
    this.cameraContext = wx.createCameraContext();
    console.log('[onReady] Camera context:', this.cameraContext ? 'Created' : 'FAILED');
    this.lastFrameTs = 0;
    this.framesSent = 0;
    this.overlayReady = false;
    this.encoderReady = false;
    this.canvasReady = false;
    this._initOverlayCanvas();
    this._initEncoderCanvas();
    this._connectSocket();
    console.log('[onReady] Complete. Overlay ready:', this.overlayReady, 'Encoder ready:', this.encoderReady);
  },
  onUnload() {
    this._stopStreaming();
    this._closeSocket();
  },
  _initOverlayCanvas() {
    wx.createSelectorQuery()
      .in(this)
      .select("#overlay")
      .fields({ node: true, size: true })
      .exec(res => {
        if (!res || !res[0]) return;
        const { node, width, height } = res[0];
        this.overlayCanvasNode = node;
        this.overlayCtx = node.getContext("2d");
        node.width = width;
        node.height = height;
        this.overlayReady = true;
        this.canvasReady = true;
      });
  },
  _initEncoderCanvas() {
    wx.createSelectorQuery()
      .in(this)
      .select("#frameEncoder")
      .fields({ node: true })
      .exec(res => {
        if (!res || !res[0]) return;
        const { node } = res[0];
        this.encoderCanvasNode = node;
        this.encoderCtx = node.getContext("2d");
        this.encoderReady = true;
      });
  },
  _connectSocket() {
    if (this.socket) return;
    // 修改为你的实际电脑局域网IP和8000端口
    const url = app?.globalData?.apiBaseUrl || "ws://192.168.1.5:8080/ws";
    this.socket = wx.connectSocket({ url });
    this.socket.onOpen(() => {
      this.setData({ connected: true });
    });
    this.socket.onClose(() => {
      this.setData({ connected: false });
      this.socket = null;
    });
    this.socket.onError(() => {
      this.setData({ connected: false });
    });
    this.socket.onMessage(evt => {
      try {
        const payload = JSON.parse(evt.data);
        this._renderDetections(payload.detections || []);
        this.setData({
          stats: {
            fps: payload.fps || this.data.stats.fps,
            detectionCount: (payload.detections || []).length
          }
        });
      } catch (err) {
        console.warn("Invalid detection payload", err);
      }
    });
  },
  _closeSocket() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  },
  _renderDetections(detections) {
    if (!this.overlayReady || !this.overlayCtx || !this.overlayCanvasNode) return;
    const ctx = this.overlayCtx;
    const canvas = this.overlayCanvasNode;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 3;
    ctx.font = "16px sans-serif";
    ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
    detections.forEach(item => {
      const [x1, y1, x2, y2] = item.bbox || [];
      const width = x2 - x1;
      const height = y2 - y1;
      ctx.strokeRect(x1, y1, width, height);
      const label = `${item.class_name || "mask"} ${(item.confidence || 0).toFixed(2)}`;
      const textWidth = ctx.measureText(label).width + 8;
      ctx.fillRect(x1, y1 - 20, textWidth, 20);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(label, x1 + 4, y1 - 6);
      ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
    });
  },
  toggleStreaming() {
    console.log('[toggleStreaming] Current streaming state:', this.data.streaming);
    console.log('[toggleStreaming] CameraContext:', this.cameraContext);
    console.log('[toggleStreaming] Canvas ready:', this.canvasReady);
    console.log('[toggleStreaming] Socket state:', this.socket ? this.socket.readyState : 'no socket');
    
    if (this.data.streaming) {
      this._stopStreaming();
    } else {
      this._startStreaming();
    }
  },
  _startStreaming() {
    console.log('[_startStreaming] Attempting to start streaming...');
    if (this.data.streaming) {
      console.log('[_startStreaming] Already streaming, skipping');
      return;
    }
    if (!this.cameraContext) {
      console.error('[_startStreaming] No camera context available');
      wx.showToast({
        title: '摄像头未初始化',
        icon: 'error'
      });
      return;
    }
    if (!this.overlayReady) {
      console.warn('[_startStreaming] Overlay canvas not ready, reinitializing...');
      this._initOverlayCanvas();
    }
    
    console.log('[_startStreaming] Setting up camera frame listener...');
    this.listener = this.cameraContext.onCameraFrame(frame => {
      const now = Date.now();
      const interval = 1000 / (app?.globalData?.frameRate || 5);
      if (now - this.lastFrameTs < interval) return;
      this.lastFrameTs = now;
      console.log('[Frame] Captured at', now, 'size:', frame.width, 'x', frame.height);
      this._sendFrame(frame);
    });
    
    this.listener.start({
      success: () => {
        console.log('[_startStreaming] Listener started successfully');
        this.setData({ streaming: true });
        wx.showToast({
          title: '检测已启动',
          icon: 'success'
        });
      },
      fail: (err) => {
        console.error('[_startStreaming] Failed to start listener:', err);
        wx.showToast({
          title: '启动失败: ' + (err.errMsg || '未知错误'),
          icon: 'error',
          duration: 3000
        });
      }
    });
  },
  _stopStreaming() {
    if (this.listener) {
      this.listener.stop();
      this.listener = null;
    }
    this.setData({ streaming: false });
  },
  _sendFrame(frame) {
    if (!this.socket || this.socket.readyState !== 1) return;
    if (!this.encoderReady || !this.encoderCanvasNode || !this.encoderCtx) {
      console.warn('[_sendFrame] Encoder canvas not ready, attempting reinitialization');
      this._initEncoderCanvas();
      return;
    }
    const { width, height, data } = frame;
    const canvas = this.encoderCanvasNode;
    canvas.width = width;
    canvas.height = height;
    const ctx = this.encoderCtx;
    const clamped = new Uint8ClampedArray(data);
    let imageData;
    if (typeof ImageData === 'function') {
      imageData = new ImageData(clamped, width, height);
    } else if (ctx && typeof ctx.createImageData === 'function') {
      imageData = ctx.createImageData(width, height);
      imageData.data.set(clamped);
    } else {
      console.error('[_sendFrame] 当前环境不支持 ImageData');
      return;
    }
    ctx.putImageData(imageData, 0, 0);
    canvas.toDataURL({
      quality: 0.8,
      success: res => {
        this.socket.send({
          data: JSON.stringify({
            type: "frame",
            payload: res.data
          })
        });
      }
    });
  },
  switchCamera() {
    const next = this.data.devicePosition === "front" ? "back" : "front";
    this.setData({ devicePosition: next });
  },
  handleCameraError(evt) {
    console.error("Camera error", evt.detail);
    wx.showToast({
      title: "摄像头不可用",
      icon: "error"
    });
  },
  handleCameraStop() {
    this._stopStreaming();
  }
});
