"use client"

import Head from 'next/head';
import { useState } from 'react';
import dynamic from 'next/dynamic';

// Import the HandAudioController dynamically to prevent SSR issues
// (Since we're using browser APIs)
const HandAudioController = dynamic(
  () => import('./components/HandAudioController'),
  { ssr: false }
);

export default function Home() {
  const [showController, setShowController] = useState(false);

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Hand Audio Controller</title>
        <meta name="description" content="Control audio with hand gestures" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto px-4 py-8">
        {!showController ? (
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl font-bold mb-6">Hand-Controlled Audio Shifter</h1>
            
            <div className="bg-white shadow-lg rounded-lg p-8 mb-8">
              <h2 className="text-2xl font-semibold mb-4">Control audio with your hands</h2>
              <p className="text-gray-700 mb-6">
                This application uses your webcam to track your hand movements and 
                adjust audio playback in real-time. Use the distance between your thumb and index 
                finger to control pitch and tempo.
              </p>
              
              <div className="grid md:grid-cols-2 gap-6 mb-8">
                <div className="bg-green-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold text-green-700 mb-2">Pitch Control</h3>
                  <p className="text-green-800">
                    Use your <strong>first hand</strong> to control pitch by adjusting the distance 
                    between your thumb and index finger.
                  </p>
                  <div className="mt-4 text-center">
                    <span className="inline-block bg-green-600 text-white rounded-full p-2 w-10 h-10">1</span>
                  </div>
                </div>
                
                <div className="bg-blue-50 p-6 rounded-lg">
                  <h3 className="text-xl font-semibold text-blue-700 mb-2">Tempo Control</h3>
                  <p className="text-blue-800">
                    Use your <strong>second hand</strong> to control tempo by adjusting the distance
                    between your thumb and index finger.
                  </p>
                  <div className="mt-4 text-center">
                    <span className="inline-block bg-blue-600 text-white rounded-full p-2 w-10 h-10">2</span>
                  </div>
                </div>
              </div>
              
              <button
                onClick={() => setShowController(true)}
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-lg transition-colors"
              >
                Launch Audio Controller
              </button>
            </div>
            
            <div className="text-gray-500 text-sm">
              <p>This application requires webcam access and will play audio.</p>
              <p>The Python backend must be running on port 5000 for this application to work.</p>
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white shadow-lg rounded-lg p-6">
              <HandAudioController />
              
              <div className="mt-6 text-center">
                <button
                  onClick={() => setShowController(false)}
                  className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-lg transition-colors"
                >
                  Back to Home
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="container mx-auto px-4 py-8">
        <div className="text-center text-gray-500 text-sm">
          <p>© {new Date().getFullYear()} Hand Audio Controller</p>
        </div>
      </footer>
    </div>
  );
}