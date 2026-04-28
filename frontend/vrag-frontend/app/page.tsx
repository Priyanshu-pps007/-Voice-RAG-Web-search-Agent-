"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

type AgentEvent =
  | { type: "status"; status: string }
  | { type: "token"; token: string }
  | { type: "done" }
  | { type: "error"; message: string };

interface BrowserSpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onend: ((event: Event) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onstart: ((event: Event) => void) | null;
  start: () => void;
  stop: () => void;
}

type SpeechRecognitionEventLike = Event & {
  resultIndex: number;
  results: SpeechRecognitionResultList;
};

type SpeechRecognitionErrorEventLike = Event & {
  error: string;
};

type SpeechRecognitionCtor = new () => BrowserSpeechRecognition;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  }
}

const DEFAULT_WS_URL =
  process.env.NEXT_PUBLIC_AGENT_WS_URL ?? "ws://127.0.0.1:8000/ws/agent";
const DEFAULT_UPLOAD_URL =
  process.env.NEXT_PUBLIC_RAG_UPLOAD_URL ?? "http://127.0.0.1:8000/rag/upload";
const MAX_UPLOAD_FILES = 5;
const BARGE_IN_RMS_THRESHOLD = 0.15;
const BARGE_IN_REQUIRED_FRAMES = 5;
const BARGE_IN_TTS_GRACE_MS = 700;

function splitSpeakableSegments(buffer: string, flushAll = false) {
  const segments: string[] = [];
  let remaining = buffer;

  while (remaining.length > 0) {
    const periodIndex = remaining.indexOf(".");
    const paragraphIndex = remaining.indexOf("\n\n");

    if (periodIndex === -1 && paragraphIndex === -1) {
      break;
    }

    const usePeriod =
      periodIndex !== -1 &&
      (paragraphIndex === -1 || periodIndex < paragraphIndex);

    const boundaryIndex = usePeriod ? periodIndex : paragraphIndex;
    const boundaryLength = usePeriod ? 1 : 2;
    const segment = remaining.slice(0, boundaryIndex + boundaryLength).trim();

    if (segment) {
      segments.push(segment);
    }

    remaining = remaining.slice(boundaryIndex + boundaryLength);
  }

  if (flushAll) {
    const finalSegment = remaining.trim();
    if (finalSegment) {
      segments.push(finalSegment);
    }
    remaining = "";
  }

  return { segments, remaining };
}

export default function Home() {
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [assistantText, setAssistantText] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [isAgentListening, setIsAgentListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [isBargeInReady, setIsBargeInReady] = useState(false);
  const [error, setError] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const shouldKeepListeningRef = useRef(false);
  const speechQueueRef = useRef<string[]>([]);
  const isListeningRef = useRef(false);
  const isSpeakingRef = useRef(false);
  const isStreamingRef = useRef(false);
  const incomingSpeechBufferRef = useRef("");
  const outgoingTranscriptRef = useRef("");
  const processSpeechQueueRef = useRef<() => void>(() => {});
  const handleAgentEventRef = useRef<(event: AgentEvent) => void>(() => {});
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const volumeFrameRef = useRef<number | null>(null);
  const loudFrameCountRef = useRef(0);
  const bargeInCooldownUntilRef = useRef(0);
  const ttsStartedAtRef = useRef(0);
  const pendingBargeInRef = useRef(false);
  const autoStartAttemptedRef = useRef(false);
  const browserSpeechSupported = useSyncExternalStore(
    () => () => {},
    () => !!(window.SpeechRecognition || window.webkitSpeechRecognition),
    () => false
  );

  const queueRecognitionResumeRef = useRef<() => void>(() => {});
  const startRecognitionNowRef = useRef<() => void>(() => {});

  const interruptCurrentResponse = useCallback(() => {
    pendingBargeInRef.current = true;
    const socket = wsRef.current;

    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ action: "interrupt" }));
    }

    if (typeof window !== "undefined") {
      window.speechSynthesis.cancel();
    }

    speechQueueRef.current = [];
    incomingSpeechBufferRef.current = "";
    isSpeakingRef.current = false;
    isStreamingRef.current = false;
    ttsStartedAtRef.current = 0;
    setIsSpeaking(false);
    setIsStreaming(false);
    setAssistantText("");
  }, []);

  const startRecognitionNow = useCallback(() => {
    if (
      !shouldKeepListeningRef.current ||
      isListeningRef.current ||
      isSpeakingRef.current ||
      isStreamingRef.current
    ) {
      return;
    }

    outgoingTranscriptRef.current = "";
    setTranscript("");
    setInterimTranscript("");

    const tryStart = () => {
      try {
        recognitionRef.current?.start();
      } catch {
        window.setTimeout(() => {
          if (
            shouldKeepListeningRef.current &&
            !isListeningRef.current &&
            !isSpeakingRef.current &&
            !isStreamingRef.current
          ) {
            tryStart();
          }
        }, 80);
      }
    };

    tryStart();
  }, []);

  const queueRecognitionResume = useCallback(() => {
    if (
      !shouldKeepListeningRef.current ||
      isSpeakingRef.current ||
      isStreamingRef.current ||
      isListeningRef.current
    ) {
      return;
    }

    window.setTimeout(() => {
      if (
        !shouldKeepListeningRef.current ||
        isSpeakingRef.current ||
        isStreamingRef.current ||
        isListeningRef.current
      ) {
        return;
      }

      try {
        outgoingTranscriptRef.current = "";
        setInterimTranscript("");
        recognitionRef.current?.start();
      } catch {
        setError("Unable to restart speech recognition automatically.");
        shouldKeepListeningRef.current = false;
        setIsAgentListening(false);
      }
    }, 250);
  }, []);

  useEffect(() => {
    queueRecognitionResumeRef.current = queueRecognitionResume;
  }, [queueRecognitionResume]);

  useEffect(() => {
    startRecognitionNowRef.current = startRecognitionNow;
  }, [startRecognitionNow]);

  const processSpeechQueue = useCallback(() => {
    if (
      typeof window === "undefined" ||
      isSpeakingRef.current ||
      speechQueueRef.current.length === 0
    ) {
      return;
    }

    const nextText = speechQueueRef.current.shift();
    if (!nextText) {
      return;
    }

    const utterance = new SpeechSynthesisUtterance(nextText);
    utterance.rate = 1;
    utterance.pitch = 1;

    utterance.onstart = () => {
      isSpeakingRef.current = true;
      ttsStartedAtRef.current = Date.now();
      setIsSpeaking(true);
      if (isListeningRef.current) {
        recognitionRef.current?.stop();
      }
    };

    utterance.onend = () => {
      isSpeakingRef.current = false;
      ttsStartedAtRef.current = 0;
      setIsSpeaking(speechQueueRef.current.length > 0);
      if (speechQueueRef.current.length > 0) {
        processSpeechQueueRef.current();
        return;
      }

      queueRecognitionResumeRef.current();
    };

    utterance.onerror = () => {
      isSpeakingRef.current = false;
      ttsStartedAtRef.current = 0;
      setIsSpeaking(false);
      if (speechQueueRef.current.length > 0) {
        processSpeechQueueRef.current();
        return;
      }

      queueRecognitionResumeRef.current();
    };

    window.speechSynthesis.speak(utterance);
  }, []);

  useEffect(() => {
    processSpeechQueueRef.current = processSpeechQueue;
  }, [processSpeechQueue]);

  const enqueueSpeech = useCallback(
    (text: string) => {
      const cleaned = text.replace(/\s+/g, " ").trim();
      if (!cleaned) {
        return;
      }

      speechQueueRef.current.push(cleaned);
      processSpeechQueue();
    },
    [processSpeechQueue]
  );

  const flushIncomingSpeech = useCallback(
    (flushAll = false) => {
      const { segments, remaining } = splitSpeakableSegments(
        incomingSpeechBufferRef.current,
        flushAll
      );

      incomingSpeechBufferRef.current = remaining;
      segments.forEach(enqueueSpeech);
    },
    [enqueueSpeech]
  );

  const handleAgentEvent = useCallback(
    (event: AgentEvent) => {
      if (event.type === "status") {
        if (event.status === "interrupted") {
          isStreamingRef.current = false;
          setIsStreaming(false);
          if (pendingBargeInRef.current) {
            startRecognitionNowRef.current();
          }
          return;
        }

        isStreamingRef.current = event.status === "started";
        setIsStreaming(event.status === "started");
        return;
      }

      if (event.type === "token") {
        setAssistantText((current) => current + event.token);
        incomingSpeechBufferRef.current += event.token;
        flushIncomingSpeech(false);
        return;
      }

      if (event.type === "done") {
        isStreamingRef.current = false;
        setIsStreaming(false);
        flushIncomingSpeech(true);
        return;
      }

      if (event.type === "error") {
        isStreamingRef.current = false;
        setIsStreaming(false);
        setError(event.message);
      }
    },
    [flushIncomingSpeech]
  );

  useEffect(() => {
    handleAgentEventRef.current = handleAgentEvent;
  }, [handleAgentEvent]);

  const connectWebSocket = useCallback(() => {
    const currentSocket = wsRef.current;
    if (
      currentSocket?.readyState === WebSocket.OPEN ||
      currentSocket?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    const socket = new WebSocket(DEFAULT_WS_URL);
    wsRef.current = socket;
    setConnectionStatus("connecting");

    socket.onopen = () => {
      setConnectionStatus("connected");
      setError("");
    };

    socket.onclose = () => {
      wsRef.current = null;
      setConnectionStatus("disconnected");
      isStreamingRef.current = false;
      setIsStreaming(false);
    };

    socket.onerror = () => {
      setConnectionStatus("error");
      setError("WebSocket connection failed.");
    };

    socket.onmessage = (message) => {
      try {
        const payload = JSON.parse(message.data) as AgentEvent;
        handleAgentEventRef.current(payload);
      } catch {
        setError("Received an unreadable response from the backend.");
      }
    };
  }, []);

  useEffect(() => {
    connectWebSocket();

    return () => {
      recognitionRef.current?.stop();
      wsRef.current?.close();
      if (volumeFrameRef.current !== null) {
        window.cancelAnimationFrame(volumeFrameRef.current);
      }
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      audioContextRef.current?.close().catch(() => {});
      if (typeof window !== "undefined") {
        window.speechSynthesis.cancel();
      }
    };
  }, [connectWebSocket]);

  const sendQuery = useCallback(
    (query: string) => {
      const trimmed = query.trim();
      if (!trimmed) {
        return;
      }

      const socket = wsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        setError("Backend socket is not connected yet.");
        connectWebSocket();
        return;
      }

      if (typeof window !== "undefined") {
        window.speechSynthesis.cancel();
      }

      speechQueueRef.current = [];
      isSpeakingRef.current = false;
      incomingSpeechBufferRef.current = "";
      setAssistantText("");
      setIsSpeaking(false);
      isStreamingRef.current = true;
      setIsStreaming(true);
      setError("");

      socket.send(JSON.stringify({ query: trimmed }));
    },
    [connectWebSocket]
  );

  const stopSpeaking = useCallback(() => {
    if (typeof window !== "undefined") {
      window.speechSynthesis.cancel();
    }
    speechQueueRef.current = [];
    isSpeakingRef.current = false;
    ttsStartedAtRef.current = 0;
    setIsSpeaking(false);
  }, []);

  const startAudioMonitor = useCallback(async () => {
    if (
      typeof window === "undefined" ||
      !navigator.mediaDevices?.getUserMedia ||
      analyserRef.current
    ) {
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const audioContext = new window.AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    mediaStreamRef.current = stream;
    audioContextRef.current = audioContext;
    analyserRef.current = analyser;
    setIsBargeInReady(true);

    const sampleBuffer = new Uint8Array(analyser.fftSize);

    const monitorVolume = () => {
      const activeAnalyser = analyserRef.current;
      if (!activeAnalyser) {
        return;
      }

      activeAnalyser.getByteTimeDomainData(sampleBuffer);

      let energy = 0;
      for (const sample of sampleBuffer) {
        const normalized = (sample - 128) / 128;
        energy += normalized * normalized;
      }

      const rms = Math.sqrt(energy / sampleBuffer.length);
      const enoughTimeSinceTtsStart =
        ttsStartedAtRef.current > 0 &&
        Date.now() - ttsStartedAtRef.current >= BARGE_IN_TTS_GRACE_MS;
      const canBargeIn =
        shouldKeepListeningRef.current &&
        isSpeakingRef.current &&
        !isListeningRef.current &&
        enoughTimeSinceTtsStart &&
        Date.now() > bargeInCooldownUntilRef.current;

      if (canBargeIn && rms >= BARGE_IN_RMS_THRESHOLD) {
        loudFrameCountRef.current += 1;
      } else {
        loudFrameCountRef.current = 0;
      }

      if (loudFrameCountRef.current >= BARGE_IN_REQUIRED_FRAMES) {
        loudFrameCountRef.current = 0;
        bargeInCooldownUntilRef.current = Date.now() + 1500;
        interruptCurrentResponse();
      }

      volumeFrameRef.current = window.requestAnimationFrame(monitorVolume);
    };

    volumeFrameRef.current = window.requestAnimationFrame(monitorVolume);
  }, [interruptCurrentResponse]);

  const uploadFiles = useCallback(async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) {
      return;
    }

    if (fileList.length > MAX_UPLOAD_FILES) {
      setError(`You can upload a maximum of ${MAX_UPLOAD_FILES} PDF files at a time.`);
      return;
    }

    const files = Array.from(fileList);
    const invalidFile = files.find(
      (file) => file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")
    );

    if (invalidFile) {
      setError(`Only PDF files are supported. Invalid file: ${invalidFile.name}`);
      return;
    }

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });

    setIsUploading(true);
    setError("");
    setUploadMessage(`Uploading ${files.length} file${files.length > 1 ? "s" : ""} to the RAG pipeline...`);
    setUploadedFiles([]);

    try {
      const response = await fetch(DEFAULT_UPLOAD_URL, {
        method: "POST",
        body: formData,
      });

      const payload = (await response.json()) as
        | {
            message?: string;
            files?: { file_name: string; chunk_count: number }[];
            detail?: string;
          }
        | undefined;

      if (!response.ok) {
        throw new Error(payload?.detail || "Upload failed.");
      }

      setUploadedFiles((payload?.files ?? []).map((file) => file.file_name));
      setUploadMessage(payload?.message ?? "Upload and ingestion completed successfully.");
    } catch (uploadError) {
      const message =
        uploadError instanceof Error ? uploadError.message : "Upload failed unexpectedly.";
      setError(message);
      setUploadMessage("");
    } finally {
      setIsUploading(false);
    }
  }, []);

  const startListening = useCallback(() => {
    const Recognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!Recognition) {
      setError("This browser does not support speech recognition.");
      return;
    }

    if (!recognitionRef.current) {
      const recognition = new Recognition();
      recognition.lang = "en-US";
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        isListeningRef.current = true;
        setIsListening(true);
        setError("");
        setInterimTranscript("");
      };

      recognition.onresult = (event: SpeechRecognitionEventLike) => {
        let finalText = "";
        let interimText = "";

        for (let index = event.resultIndex; index < event.results.length; index += 1) {
          const result = event.results[index];
          const text = result[0]?.transcript ?? "";

          if (result.isFinal) {
            finalText += text;
          } else {
            interimText += text;
          }
        }

        if (finalText) {
          outgoingTranscriptRef.current = [
            outgoingTranscriptRef.current.trim(),
            finalText.trim(),
          ]
            .filter(Boolean)
            .join(" ");
          setTranscript(outgoingTranscriptRef.current);
        }

        setInterimTranscript(interimText.trim());
      };

      recognition.onerror = (event: SpeechRecognitionErrorEventLike) => {
        isListeningRef.current = false;
        setIsListening(false);
        if (event.error !== "aborted") {
          setError(`Speech recognition error: ${event.error}`);
        }
      };

      recognition.onend = () => {
        isListeningRef.current = false;
        setIsListening(false);
        setInterimTranscript("");
        const finalQuery = outgoingTranscriptRef.current.trim();
        if (finalQuery) {
          pendingBargeInRef.current = false;
          if (isStreamingRef.current || isSpeakingRef.current) {
            interruptCurrentResponse();
          }
          sendQuery(finalQuery);
          outgoingTranscriptRef.current = "";
        } else if (pendingBargeInRef.current) {
          startRecognitionNowRef.current();
        }

        queueRecognitionResumeRef.current();
      };

      recognitionRef.current = recognition;
    }

    shouldKeepListeningRef.current = true;
    setIsAgentListening(true);
    setTranscript("");
    setInterimTranscript("");
    outgoingTranscriptRef.current = "";
    void startAudioMonitor()
      .then(() => {
        recognitionRef.current?.start();
      })
      .catch(() => {
        setError("Microphone access is required for continuous listening.");
      });
  }, [interruptCurrentResponse, sendQuery, startAudioMonitor]);

  const stopListening = useCallback(() => {
    shouldKeepListeningRef.current = false;
    isListeningRef.current = false;
    setIsAgentListening(false);
    recognitionRef.current?.stop();
    setIsListening(false);
  }, []);

  useEffect(() => {
    if (!browserSpeechSupported || autoStartAttemptedRef.current || isAgentListening) {
      return;
    }

    autoStartAttemptedRef.current = true;
    startListening();
  }, [browserSpeechSupported, isAgentListening, startListening]);

  const resetSession = useCallback(() => {
    setTranscript("");
    setInterimTranscript("");
    setAssistantText("");
    setError("");
    incomingSpeechBufferRef.current = "";
    outgoingTranscriptRef.current = "";
    speechQueueRef.current = [];
    setIsStreaming(false);
    stopSpeaking();
  }, [stopSpeaking]);

  return (
    <main className="voice-shell px-4 py-6 text-stone-50 sm:px-6 lg:px-8">
      <div className="voice-orb voice-orb-left" />
      <div className="voice-orb voice-orb-right" />
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
        <section className="hero-panel">
          <div className="hero-copy">
            <p className="eyebrow">Voice RAG Concierge</p>
            <h1 className="hero-title">A calmer, sharper voice workspace for your documents and web research.</h1>
            <p className="hero-text">
              The agent listens by default, yields when you interrupt, and speaks back in buffered, low-latency turns.
            </p>
            <div className="hero-tags">
              <span className="hero-tag">Always-on listening</span>
              <span className="hero-tag">Barge-in aware</span>
              <span className="hero-tag">Private document RAG</span>
            </div>
          </div>
          <div className="hero-status-wrap">
            <div
              className={`status-pill ${
                connectionStatus === "connected"
                  ? "status-online"
                  : connectionStatus === "connecting"
                    ? "status-waiting"
                    : "status-offline"
              }`}
            >
              {connectionStatus}
            </div>
            <div className="presence-card">
              <span className="presence-dot" />
              <div>
                <p className="panel-label">Agent Presence</p>
                <p className="presence-text">
                  {isAgentListening ? "Listening in the background" : "Paused until resumed"}
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="stats-grid">
          <div className="metric-box">
            <span className="metric-label">Speech Input</span>
            <span className="metric-value">{browserSpeechSupported ? "Ready" : "Unavailable"}</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Barge-In</span>
            <span className="metric-value">{isBargeInReady ? "Active" : "Needs mic"}</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Live Mic</span>
            <span className="metric-value">{isListening ? "Capturing" : "Standby"}</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Agent Voice</span>
            <span className="metric-value">{isSpeaking ? "Speaking" : "Waiting"}</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Streaming</span>
            <span className="metric-value">{isStreaming ? "In flight" : "Idle"}</span>
          </div>
          <div className="metric-box">
            <span className="metric-label">Document Memory</span>
            <span className="metric-value">{uploadedFiles.length ? `${uploadedFiles.length} recent` : "Ready"}</span>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
          <div className="panel">
            <div className="section-head">
              <div>
                <p className="panel-label">Document Intake</p>
                <h2 className="section-title">Drop new PDFs into the agent’s memory</h2>
              </div>
              <button className="secondary-button" onClick={connectWebSocket} type="button">
                Reconnect
              </button>
            </div>

            <label className="upload-stage" htmlFor="pdf-upload">
              <input
                id="pdf-upload"
                accept="application/pdf,.pdf"
                className="sr-only"
                disabled={isUploading}
                multiple
                onChange={(event) => {
                  void uploadFiles(event.target.files);
                  event.currentTarget.value = "";
                }}
                type="file"
              />
              <span className="upload-badge">{isUploading ? "Processing" : "Upload PDFs"}</span>
              <span className="upload-title">
                {isUploading ? "Running the RAG pipeline for your files" : "Drag files here or click to browse"}
              </span>
              <span className="upload-copy">
                Up to {MAX_UPLOAD_FILES} PDFs at a time. Each file is chunked, embedded, and made searchable before the agent uses it.
              </span>
            </label>

            {uploadMessage ? (
              <div className="upload-feedback">
                <p>{uploadMessage}</p>
                {uploadedFiles.length ? (
                  <div className="chip-row">
                    {uploadedFiles.map((file) => (
                      <span key={file} className="file-chip">
                        {file}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}

            <div className="control-strip">
              <button
                className="secondary-button"
                disabled={!isAgentListening && !isListening}
                onClick={stopListening}
                type="button"
              >
                Pause ears
              </button>
              <button
                className="secondary-button"
                disabled={!browserSpeechSupported || isAgentListening}
                onClick={startListening}
                type="button"
              >
                Resume listening
              </button>
              <button
                className="secondary-button"
                disabled={!isSpeaking}
                onClick={stopSpeaking}
                type="button"
              >
                Silence voice
              </button>
              <button className="secondary-button" onClick={resetSession} type="button">
                Reset session
              </button>
            </div>
          </div>

          <div className="grid gap-6">
            <div className="panel">
              <div className="section-head">
                <div>
                  <p className="panel-label">Live Conversation</p>
                  <h2 className="section-title">What the agent is hearing right now</h2>
                </div>
              </div>

              <div className="conversation-stack">
                <div>
                  <p className="panel-label">Final Transcript</p>
                  <div className="transcript-box transcript-primary">
                    {transcript || "Start speaking. The agent is already listening in the background."}
                  </div>
                </div>
                <div>
                  <p className="panel-label">Live Partial</p>
                  <div className="transcript-box transcript-muted">
                    {interimTranscript || "Partial speech will appear here as you talk."}
                  </div>
                </div>
              </div>
            </div>

            <div className="panel assistant-panel">
              <div className="section-head">
                <div>
                  <p className="panel-label">Assistant Stream</p>
                  <h2 className="section-title">Buffered answer playback</h2>
                </div>
                <span className="stream-pill">Sentence-based TTS</span>
              </div>

              <div className="transcript-box transcript-assistant whitespace-pre-wrap">
                {assistantText || "The assistant’s streamed response will land here and speak sentence by sentence."}
              </div>
            </div>
          </div>
        </section>

        {error ? (
          <section className="error-panel">
            {error}
          </section>
        ) : null}
      </div>
    </main>
  );
}
