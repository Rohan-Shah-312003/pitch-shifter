// Create a placeholder-camera.png in your public folder
// You can add this with a simple SVG or PNG file showing a camera icon
// Or use the following code to create a simple camera placeholder

// In hand-audio-frontend/app/components/CameraPlaceholder.jsx
import React from 'react';

export default function CameraPlaceholder() {
  return (
    <svg 
      width="640" 
      height="480" 
      viewBox="0 0 640 480" 
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="640" height="480" fill="#f0f0f0" />
      <g transform="translate(270, 190)">
        <path 
          d="M50,0 L70,20 L100,20 L100,80 L0,80 L0,20 L30,20 Z" 
          fill="#888888" 
          stroke="#666666" 
          strokeWidth="2"
        />
        <circle cx="50" cy="50" r="20" fill="#aaaaaa" stroke="#666666" strokeWidth="2" />
      </g>
      <text 
        x="320" 
        y="300" 
        fontFamily="Arial" 
        fontSize="16" 
        textAnchor="middle" 
        fill="#666666"
      >
        Camera Feed Placeholder
      </text>
    </svg>
  );
}