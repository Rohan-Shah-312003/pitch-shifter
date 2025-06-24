import '../styles/globals.css'
import { useEffect } from 'react'

export default function App({ Component, pageProps }) {
  useEffect(() => {
    // Check if backend is running
    const checkBackend = async () => {
      try {
        const response = await fetch(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/health');
        if (!response.ok) {
          console.warn('Backend server is not responding. Please make sure the Python backend is running on port 5000.');
        }
      } catch (error) {
        console.warn('Could not connect to backend server. Please make sure the Python backend is running on port 5000.');
      }
    };

    checkBackend();
  }, []);

  return <Component {...pageProps} />
}