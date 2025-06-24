# Hand Gesture Audio Controller

A real-time audio manipulation system that uses computer vision to track hand gestures and modify audio pitch and tempo accordingly.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe to detect hand landmarks
- **Audio Manipulation**: 
  - Left hand controls pitch (0.5x to 2x)
  - Right hand controls tempo (0.5x to 2x)
- **Live Video Feed**: Real-time camera stream with hand landmark visualization
- **Real-time Visualizations**: Live plots showing audio parameters
- **Modern UI**: Responsive Next.js frontend with glassmorphism design

## Architecture

- **Backend**: Python Flask server with computer vision and audio processing
- **Frontend**: Next.js React application with real-time data streaming
- **Communication**: RESTful API with real-time polling for live updates

## Prerequisites

- Python 3.8+
- Node.js 16+
- Webcam
- Audio files (WAV, MP3, etc.)

## Installation

### Backend Setup

1. Navigate to the backend directory and install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Python backend:
```bash
python hand_audio_controller.py
```

The backend will start on `http://localhost:5000`

### Frontend Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. **Upload Audio**: Click "Choose Audio File" to upload your audio file
2. **Start Camera**: Click "Start Camera" to begin hand tracking
3. **Start Audio**: Click "Start Audio" to begin playback
4. **Control Audio**:
   - **Left Hand**: Move thumb and index finger closer/farther to control pitch
   - **Right Hand**: Move thumb and index finger closer/farther to control tempo

## Hand Gesture Controls

### Left Hand (Pitch Control)
- **Close fingers** (small distance) = Lower pitch (0.5x)
- **Spread fingers** (large distance) = Higher pitch (2.0x)

### Right Hand (Tempo Control)
- **Close fingers** (small distance) = Slower tempo (0.5x)
- **Spread fingers** (large distance) = Faster tempo (2.0x)

## API Endpoints

### Backend Endpoints
- `POST /upload_audio` - Upload audio file
- `POST /start_audio` - Start audio playback
- `POST /stop_audio` - Stop audio playback
- `POST /start_camera` - Start camera capture
- `POST /stop_camera` - Stop camera capture
- `GET /health` - Health check endpoint

## File Structure

```
hand-audio-controller/
├── backend/
│   ├── hand_audio_controller.py    # Main Python backend
│   └── requirements.txt            # Python dependencies
├── frontend/
│   ├── pages/
│   │   ├── index.js               # Main React component
│   │   └── _app.js                # Next.js app wrapper
│   ├── styles/
│   │   └── globals.css            # Global CSS styles
│   ├── package.json               # Node.js dependencies
│   ├── tailwind.config.js         # Tailwind configuration
│   └── .env.local                 # Environment variables
└── README.md                      # This file
```

## Technical Details

### Backend Technologies
- **Flask**: Web framework for API endpoints
- **MediaPipe**: Hand landmark detection
- **OpenCV**: Computer vision and video processing
- **librosa**: Audio analysis and modification
- **pygame**: Audio playback
- **matplotlib**: Real-time visualization plots

### Frontend Technologies
- **Next.js**: React framework
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Icon library
- **Real-time polling**: Live data updates

### Audio Processing
- **Pitch Shifting**: Uses librosa's pitch_shift with semitone calculations
- **Time Stretching**: Uses librosa's time_stretch for tempo changes
- **Real-time Processing**: Continuous audio modification based on hand gestures

### Computer Vision
- **Hand Detection**: MediaPipe Hands solution
- **Landmark Tracking**: 21 hand landmarks per hand
- **Distance Calculation**: Euclidean distance between thumb tip and index finger tip
- **Gesture Mapping**: Linear mapping from finger distance to audio parameters

## Customization

### Adjusting Sensitivity
In `hand_audio_controller.py`, modify these parameters:

```python
# Distance ranges for mapping
self.min_distance = 0.02  # Minimum finger distance
self.max_distance = 0.15  # Maximum finger distance
self.pitch_range = (0.5, 2.0)  # Pitch multiplier range
self.tempo_range = (0.5, 2.0)  # Tempo multiplier range
```

### UI Customization
- Modify `tailwind.config.js` for custom themes
- Update `globals.css` for custom animations
- Edit `pages/index.js` for layout changes

## Deployment

### Backend Deployment
1. Use a cloud service like Heroku, AWS, or DigitalOcean
2. Install dependencies and run the Flask app
3. Ensure camera access is available (may require specific server setup)

### Frontend Deployment
1. Build the Next.js app:
```bash
npm run build
```

2. Deploy to Vercel, Netlify, or your preferred hosting service
3. Update `NEXT_PUBLIC_API_URL` in environment variables

## Troubleshooting

### Common Issues

1. **Camera not working**:
   - Check browser permissions for camera access
   - Ensure no other applications are using the camera
   - Try different browsers (Chrome recommended)

2. **Audio not playing**:
   - Check audio file format compatibility
   - Ensure audio file is not corrupted
   - Check system audio settings

3. **Hand tracking not working**:
   - Ensure good lighting conditions
   - Keep hands visible in camera frame
   - Avoid background clutter

4. **Backend connection issues**:
   - Verify Python backend is running on port 5000
   - Check firewall settings
   - Ensure all dependencies are installed

### Performance Optimization

1. **Reduce polling frequency** in frontend for slower devices
2. **Adjust audio buffer size** in pygame initialization
3. **Lower camera resolution** for better performance
4. **Optimize visualization update rate**

## Browser Compatibility

- **Chrome**: Recommended (full WebRTC support)
- **Firefox**: Supported
- **Safari**: Limited (may have camera issues)
- **Edge**: Supported

## Security Considerations

- Audio files are processed locally and not stored permanently
- Camera feed is processed in real-time without recording
- All data processing happens locally on your machine

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- MediaPipe team for hand tracking technology
- librosa developers for audio processing capabilities
- Next.js and React communities
- Tailwind CSS for the utility-first CSS framework

## Future Enhancements

- [ ] Support for more audio formats
- [ ] Multiple audio track mixing
- [ ] Gesture recording and playback
- [ ] MIDI controller output
- [ ] Advanced audio effects (reverb, echo, etc.)
- [ ] Mobile app version
- [ ] Multi-user collaboration features
- [ ] Machine learning for custom gesture recognition

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the GitHub issues
3. Create a new issue with detailed information about your problem

---

**Happy gesture controlling!** 🎵✋ /video_feed` - Live camera stream
- `GET /get_data` - Current hand tracking data
- `GET /get_visualization` - Real-time visualization plots
- `GET