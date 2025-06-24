import React, { useState, useEffect, useRef } from 'react';
import Head from 'next/head';
import { Upload, Play, Pause, Camera, CameraOff, Activity, Volume2, Clock } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export default function HandAudioController() {
  // State management
  const [audioFile, setAudioFile] = useState(null);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [isCameraRunning, setIsCameraRunning] = useState(false);
  const [currentData, setCurrentData] = useState({
    left_hand_distance: 0,
    right_hand_distance: 0,
    pitch_factor: 1.0,
    tempo_factor: 1.0,
    audio_playing: false,
    camera_running: false
  });
  const [visualizationImage, setVisualizationImage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Refs
  const fileInputRef = useRef(null);
  const dataIntervalRef = useRef(null);
  const visualizationIntervalRef = useRef(null);

  // Clear messages after 3 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(''), 3000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(''), 3000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  // Start/stop data polling
  useEffect(() => {
    if (isCameraRunning || isAudioPlaying) {
      startDataPolling();
      startVisualizationPolling();
    } else {
      stopDataPolling();
      stopVisualizationPolling();
    }

    return () => {
      stopDataPolling();
      stopVisualizationPolling();
    };
  }, [isCameraRunning, isAudioPlaying]);

  const startDataPolling = () => {
    if (dataIntervalRef.current) return;
    
    dataIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/get_data`);
        if (response.ok) {
          const data = await response.json();
          setCurrentData(data);
        }
      } catch (err) {
        console.error('Error fetching data:', err);
      }
    }, 100); // 10 FPS data polling
  };

  const stopDataPolling = () => {
    if (dataIntervalRef.current) {
      clearInterval(dataIntervalRef.current);
      dataIntervalRef.current = null;
    }
  };

  const startVisualizationPolling = () => {
    if (visualizationIntervalRef.current) return;
    
    visualizationIntervalRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/get_visualization`);
        if (response.ok) {
          const data = await response.json();
          setVisualizationImage(data.plot);
        }
      } catch (err) {
        console.error('Error fetching visualization:', err);
      }
    }, 500); // 2 FPS visualization polling
  };

  const stopVisualizationPolling = () => {
    if (visualizationIntervalRef.current) {
      clearInterval(visualizationIntervalRef.current);
      visualizationIntervalRef.current = null;
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setAudioFile(file);
    setIsLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('audio', file);

      const response = await fetch(`${API_BASE_URL}/upload_audio`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setSuccess('Audio file uploaded successfully!');
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Failed to upload audio');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleAudio = async () => {
    if (!audioFile) {
      setError('Please upload an audio file first');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const endpoint = isAudioPlaying ? '/stop_audio' : '/start_audio';
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
      });

      if (response.ok) {
        setIsAudioPlaying(!isAudioPlaying);
        setSuccess(isAudioPlaying ? 'Audio stopped' : 'Audio started');
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Failed to control audio');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleCamera = async () => {
    setIsLoading(true);
    setError('');

    try {
      const endpoint = isCameraRunning ? '/stop_camera' : '/start_camera';
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
      });

      if (response.ok) {
        setIsCameraRunning(!isCameraRunning);
        setSuccess(isCameraRunning ? 'Camera stopped' : 'Camera started');
      } else {
        const errorData = await response.json();
        setError(errorData.error || 'Failed to control camera');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (value, min = 0.5, max = 2.0) => {
    const normalized = (value - min) / (max - min);
    if (normalized < 0.33) return 'text-blue-500';
    if (normalized < 0.66) return 'text-green-500';
    return 'text-red-500';
  };

  const getProgressWidth = (value, min = 0.5, max = 2.0) => {
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    return `${normalized * 100}%`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900">
      <Head>
        <title>Hand Gesture Audio Controller</title>
        <meta name="description" content="Control audio pitch and tempo with hand gestures" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">
            Hand Gesture Audio Controller
          </h1>
          <p className="text-gray-300 text-lg">
            Control audio pitch with your left hand and tempo with your right hand
          </p>
        </div>

        {/* Status Messages */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}
        {success && (
          <div className="mb-6 p-4 bg-green-500/20 border border-green-500 rounded-lg text-green-200">
            {success}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Controls */}
          <div className="space-y-6">
            {/* File Upload */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Upload className="mr-2" size={20} />
                Audio Upload
              </h2>
              <div className="space-y-4">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  className="w-full p-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg text-white font-medium transition-colors"
                >
                  {isLoading ? 'Uploading...' : 'Choose Audio File'}
                </button>
                {audioFile && (
                  <p className="text-gray-300 text-sm">
                    Selected: {audioFile.name}
                  </p>
                )}
              </div>
            </div>

            {/* Control Buttons */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4">Controls</h2>
              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={toggleAudio}
                  disabled={isLoading || !audioFile}
                  className={`p-4 rounded-lg font-medium transition-colors flex items-center justify-center ${
                    isAudioPlaying
                      ? 'bg-red-600 hover:bg-red-700'
                      : 'bg-green-600 hover:bg-green-700'
                  } disabled:bg-gray-600 text-white`}
                >
                  {isAudioPlaying ? (
                    <>
                      <Pause className="mr-2" size={20} />
                      Stop Audio
                    </>
                  ) : (
                    <>
                      <Play className="mr-2" size={20} />
                      Start Audio
                    </>
                  )}
                </button>

                <button
                  onClick={toggleCamera}
                  disabled={isLoading}
                  className={`p-4 rounded-lg font-medium transition-colors flex items-center justify-center ${
                    isCameraRunning
                      ? 'bg-red-600 hover:bg-red-700'
                      : 'bg-blue-600 hover:bg-blue-700'
                  } disabled:bg-gray-600 text-white`}
                >
                  {isCameraRunning ? (
                    <>
                      <CameraOff className="mr-2" size={20} />
                      Stop Camera
                    </>
                  ) : (
                    <>
                      <Camera className="mr-2" size={20} />
                      Start Camera
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Real-time Data */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Activity className="mr-2" size={20} />
                Real-time Data
              </h2>
              <div className="space-y-4">
                {/* Pitch Control */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-300 flex items-center">
                      <Volume2 className="mr-2" size={16} />
                      Pitch Factor (Left Hand)
                    </span>
                    <span className={`font-mono ${getStatusColor(currentData.pitch_factor)}`}>
                      {currentData.pitch_factor.toFixed(2)}x
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: getProgressWidth(currentData.pitch_factor) }}
                    />
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    Distance: {currentData.left_hand_distance.toFixed(3)}
                  </div>
                </div>

                {/* Tempo Control */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-300 flex items-center">
                      <Clock className="mr-2" size={16} />
                      Tempo Factor (Right Hand)
                    </span>
                    <span className={`font-mono ${getStatusColor(currentData.tempo_factor)}`}>
                      {currentData.tempo_factor.toFixed(2)}x
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-green-500 to-yellow-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: getProgressWidth(currentData.tempo_factor) }}
                    />
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    Distance: {currentData.right_hand_distance.toFixed(3)}
                  </div>
                </div>
              </div>
            </div>

            {/* Instructions */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4">How to Use</h2>
              <div className="space-y-3 text-gray-300">
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold mr-3 mt-0.5">
                    1
                  </div>
                  <p>Upload an audio file using the upload button</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold mr-3 mt-0.5">
                    2
                  </div>
                  <p>Start the camera to enable hand tracking</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold mr-3 mt-0.5">
                    3
                  </div>
                  <p>Start audio playback</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold mr-3 mt-0.5">
                    4
                  </div>
                  <p>
                    <strong>Left hand:</strong> Control pitch by changing distance between thumb and index finger
                  </p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold mr-3 mt-0.5">
                    5
                  </div>
                  <p>
                    <strong>Right hand:</strong> Control tempo by changing distance between thumb and index finger
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Video and Visualization */}
          <div className="space-y-6">
            {/* Camera Feed */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4">Camera Feed</h2>
              <div className="aspect-video bg-gray-800 rounded-lg overflow-hidden">
                {isCameraRunning ? (
                  <img
                    src={`${API_BASE_URL}/video_feed`}
                    alt="Camera feed"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-400">
                    <div className="text-center">
                      <Camera size={64} className="mx-auto mb-4 opacity-50" />
                      <p>Camera not started</p>
                      <p className="text-sm">Click "Start Camera" to begin hand tracking</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Visualization */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4">Real-time Visualization</h2>
              <div className="bg-white rounded-lg p-4">
                {visualizationImage ? (
                  <img
                    src={`data:image/png;base64,${visualizationImage}`}
                    alt="Real-time visualization"
                    className="w-full h-auto"
                  />
                ) : (
                  <div className="aspect-video flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <Activity size={64} className="mx-auto mb-4 opacity-50" />
                      <p>No visualization data</p>
                      <p className="text-sm">Start camera and audio to see live plots</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center">
          <p className="text-gray-400">
            Hand Gesture Audio Controller - Control audio with intuitive hand movements
          </p>
        </div>
      </div>
    </div>
  );
}