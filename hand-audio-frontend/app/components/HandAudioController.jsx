import { useState, useEffect, useRef } from 'react';

export default function HandAudioController() {
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("🤚 Not running");
  const [pitch, setPitch] = useState(0);
  const [rate, setRate] = useState(1.0);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState(null);
  const [audioError, setAudioError] = useState(null);
  
  const frameRef = useRef(null);
  const frameTimerRef = useRef(null);
  const retryCountRef = useRef(0);
  
  // API endpoint configuration
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
  
  // Start processing
  const startProcessing = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setAudioError(null);
      
      const response = await fetch(`${API_BASE}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      if (data.status === 'started' || data.status === 'already_running') {
        setIsRunning(true);
        startFrameFetching();
      } else if (data.status === 'error') {
        setError(data.message || "Failed to start processing");
      } else {
        setError("Failed to start processing");
      }
    } catch (err) {
      setError(`Error connecting to backend: ${err.message}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Stop processing
  const stopProcessing = async () => {
    try {
      setIsLoading(true);
      
      // Stop frame fetching
      if (frameTimerRef.current) {
        clearInterval(frameTimerRef.current);
        frameTimerRef.current = null;
      }
      
      const response = await fetch(`${API_BASE}/api/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      if (data.status === 'stopped' || data.status === 'not_running') {
        setIsRunning(false);
        setStatusMessage("🤚 Not running");
        setPitch(0);
        setRate(1.0);
        setFps(0);
        setAudioError(null);
      }
    } catch (err) {
      setError(`Error stopping backend: ${err.message}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Start periodic frame fetching
  const startFrameFetching = () => {
    // Clear any existing timer
    if (frameTimerRef.current) {
      clearInterval(frameTimerRef.current);
    }
    
    // Start a new timer to fetch frames
    frameTimerRef.current = setInterval(fetchCurrentFrame, 100); // 10 fps for UI updates
  };
  
  // Fetch the current frame
  const fetchCurrentFrame = async () => {
    if (!isRunning) return;
    
    try {
      const response = await fetch(`${API_BASE}/api/frame`);
      
      if (!response.ok) {
        // Don't set error on failed frame, just log it
        if (response.status !== 404) { // Ignore "No frames available"
          console.error(`Error fetching frame: ${response.status}`);
          
          // After several failed attempts, consider the backend down
          retryCountRef.current += 1;
          if (retryCountRef.current > 50) { // After ~5 seconds of failures
            setError("Connection to backend lost");
            setIsRunning(false);
            clearInterval(frameTimerRef.current);
          }
        }
        return;
      }
      
      // Reset retry counter on successful fetch
      retryCountRef.current = 0;
      
      const data = await response.json();
      
      // Update state with data from backend
      setStatusMessage(data.status);
      setPitch(data.pitch);
      setRate(data.rate);
      setFps(data.fps);
      
      // Check for audio errors
      if (data.audio_error) {
        setAudioError(data.audio_error);
      } else {
        setAudioError(null);
      }
      
      // Update the image
      if (data.frame) {
        frameRef.current.src = `data:image/jpeg;base64,${data.frame}`;
      }
    } catch (err) {
      console.error("Error fetching frame:", err);
      // Increment retry count but don't immediately stop on network errors
      retryCountRef.current += 1;
    }
  };
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (frameTimerRef.current) {
        clearInterval(frameTimerRef.current);
      }
      
      // Stop the backend processing if component unmounts while running
      if (isRunning) {
        fetch(`${API_BASE}/api/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        }).catch(err => console.error("Error stopping on unmount:", err));
      }
    };
  }, [isRunning, API_BASE]);
  
  return (
    <div className="flex flex-col items-center w-full max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Hand-Controlled Audio Shifter</h1>
      
      <div className="relative w-full aspect-video bg-gray-100 border-2 border-gray-300 rounded-lg overflow-hidden mb-4">
        {!isRunning && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-40 text-white text-center p-6">
            <div>
              <p className="text-xl font-bold mb-2">Camera Feed Not Active</p>
              <p>Click Start to begin hand tracking and audio control</p>
            </div>
          </div>
        )}
        
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-white mb-2"></div>
              <p>{isRunning ? "Stopping..." : "Starting..."}</p>
            </div>
          </div>
        )}
        
        <img 
          ref={frameRef}
          className="w-full h-full object-contain"
          alt="Camera feed"
          src="/placeholder-camera.png"
        />
      </div>
      
      {error && (
        <div className="w-full bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <p>{error}</p>
        </div>
      )}
      
      {audioError && (
        <div className="w-full bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded mb-4">
          <p className="font-bold">Audio Issue:</p>
          <p>{audioError}</p>
        </div>
      )}
      
      <div className="grid grid-cols-3 gap-4 w-full mb-6">
        <div className="bg-gray-100 p-4 rounded-lg text-center">
          <h2 className="font-semibold mb-1">Status</h2>
          <p className="text-lg">{statusMessage}</p>
        </div>
        <div className="bg-gray-100 p-4 rounded-lg text-center">
          <h2 className="font-semibold mb-1">Pitch</h2>
          <p className="text-lg font-bold text-green-600">{pitch.toFixed(2)}</p>
        </div>
        <div className="bg-gray-100 p-4 rounded-lg text-center">
          <h2 className="font-semibold mb-1">Tempo</h2>
          <p className="text-lg font-bold text-blue-600">{rate.toFixed(2)}</p>
        </div>
      </div>
      
      <div className="flex items-center justify-center mb-6">
        <button
          onClick={isRunning ? stopProcessing : startProcessing}
          disabled={isLoading}
          className={`
            px-6 py-3 rounded-lg font-semibold text-white
            ${isLoading ? 'bg-gray-400 cursor-not-allowed' : 
              isRunning ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'}
            transition-colors
          `}
        >
          {isRunning ? 'Stop Controller' : 'Start Controller'}
        </button>
        {isRunning && (
          <div className="ml-4 text-sm text-gray-500">
            FPS: {fps}
          </div>
        )}
      </div>
      
      <div className="w-full bg-yellow-50 border-l-4 border-yellow-400 p-4 text-sm">
        <h3 className="font-bold text-yellow-800">How to use:</h3>
        <ul className="mt-2 list-disc pl-5 text-yellow-700 space-y-1">
          <li><span className="font-semibold">First hand</span>: Control pitch with the distance between thumb and index finger</li>
          <li><span className="font-semibold">Second hand</span>: Control tempo with the distance between thumb and index finger</li>
          <li>Wider distances increase values, closer distances decrease values</li>
          {audioError && (
            <li className="text-red-600 font-bold">Note: There is an issue with audio playback. The visual hand tracking will still work.</li>
          )}
        </ul>
      </div>
      
      <div className="w-full mt-4 text-gray-600 text-sm">
        <p className="text-center">
          If you experience audio issues, try restarting the application or checking your audio device settings.
        </p>
      </div>
    </div>
  );
}