import numpy as np
import pyaudio
import threading
import time
import scipy.signal
from scipy import fft
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import librosa
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class NoiseOptimizer:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        
        # Analysis parameters
        self.noise_buffer = deque(maxlen=100)  # Store recent ambient analysis
        self.is_running = False
        
        # Noise generation parameters
        self.current_mode = "focus"  # focus, sleep, adaptive
        self.noise_types = {
            'white': self._generate_white_noise,
            'pink': self._generate_pink_noise,
            'brown': self._generate_brown_noise,
            'nature': self._generate_nature_sounds,
            'binaural': self._generate_binaural_beats
        }
        
        # Adaptive parameters
        self.user_profile = self._load_user_profile()
        self.ambient_classifier = AmbientSoundClassifier()
        
        # EEG simulation (replace with real EEG interface)
        self.eeg_simulator = EEGSimulator()
        
    def _load_user_profile(self):
        """Load or create user profile with preferences"""
        profile_file = "user_profile.json"
        if os.path.exists(profile_file):
            with open(profile_file, 'r') as f:
                return json.load(f)
        else:
            # Default profile
            profile = {
                'preferred_frequencies': {'focus': [40, 60], 'sleep': [0.5, 4]},
                'noise_preferences': {'focus': 'pink', 'sleep': 'brown'},
                'sensitivity_level': 0.5,
                'adaptation_rate': 0.1
            }
            self._save_user_profile(profile)
            return profile
    
    def _save_user_profile(self, profile):
        """Save user profile"""
        with open("user_profile.json", 'w') as f:
            json.dump(profile, f, indent=2)
    
    def start_optimization(self, mode="adaptive"):
        """Start the noise optimization system"""
        self.current_mode = mode
        self.is_running = True
        
        # Initialize audio streams
        self._setup_audio_streams()
        
        # Start threads
        self.analysis_thread = threading.Thread(target=self._ambient_analysis_loop)
        self.generation_thread = threading.Thread(target=self._noise_generation_loop)
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop)
        
        self.analysis_thread.start()
        self.generation_thread.start()
        self.adaptation_thread.start()
        
        print(f"🎧 Noise Optimizer started in {mode} mode")
        print("Press Ctrl+C to stop...")
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_optimization()
    
    def stop_optimization(self):
        """Stop the optimization system"""
        print("\n🛑 Stopping Noise Optimizer...")
        self.is_running = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        self.audio.terminate()
        print("✅ Stopped successfully")
    
    def _setup_audio_streams(self):
        """Setup input and output audio streams"""
        try:
            # Input stream for ambient sound analysis
            self.input_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Output stream for generated noise
            self.output_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=2,  # Stereo for binaural beats
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
        except Exception as e:
            print(f"❌ Audio setup failed: {e}")
            print("💡 Make sure your microphone and speakers are properly connected")
    
    def _ambient_analysis_loop(self):
        """Continuously analyze ambient sound"""
        while self.is_running:
            try:
                # Read audio data
                data = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # Analyze ambient sound
                analysis = self._analyze_ambient_sound(audio_data)
                self.noise_buffer.append(analysis)
                
                # Print real-time analysis
                if len(self.noise_buffer) % 10 == 0:  # Every 10 samples
                    self._print_analysis_summary()
                
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Analysis error: {e}")
            
            time.sleep(0.1)
    
    def _analyze_ambient_sound(self, audio_data):
        """Analyze ambient sound characteristics"""
        # Spectral analysis
        freqs, psd = scipy.signal.welch(audio_data, self.sample_rate)
        
        # Feature extraction
        analysis = {
            'timestamp': time.time(),
            'rms_level': np.sqrt(np.mean(audio_data**2)),
            'spectral_centroid': np.sum(freqs * psd) / np.sum(psd),
            'spectral_rolloff': self._calculate_spectral_rolloff(freqs, psd),
            'zero_crossing_rate': self._calculate_zcr(audio_data),
            'dominant_frequencies': freqs[np.argsort(psd)[-5:]],
            'noise_floor': np.percentile(psd, 10),
            'environment_type': self.ambient_classifier.classify(audio_data)
        }
        
        return analysis
    
    def _calculate_spectral_rolloff(self, freqs, psd, rolloff_percent=0.85):
        """Calculate spectral rolloff frequency"""
        cumulative_sum = np.cumsum(psd)
        rolloff_threshold = rolloff_percent * cumulative_sum[-1]
        rolloff_idx = np.where(cumulative_sum >= rolloff_threshold)[0][0]
        return freqs[rolloff_idx]
    
    def _calculate_zcr(self, audio_data):
        """Calculate zero crossing rate"""
        return np.sum(np.diff(np.sign(audio_data)) != 0) / (2 * len(audio_data))
    
    def _noise_generation_loop(self):
        """Generate and output optimized noise"""
        phase = 0
        while self.is_running:
            try:
                # Generate noise based on current analysis
                noise_type = self._determine_optimal_noise_type()
                noise_params = self._calculate_noise_parameters()
                
                # Generate audio chunk
                audio_chunk = self._generate_adaptive_noise(noise_type, noise_params, phase)
                phase += self.chunk_size / self.sample_rate
                
                # Output audio
                if self.output_stream:
                    self.output_stream.write(audio_chunk.tobytes())
                
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Generation error: {e}")
            
            time.sleep(0.01)
    
    def _adaptation_loop(self):
        """Continuously adapt to user's state and environment"""
        while self.is_running:
            try:
                # Simulate EEG reading (replace with real EEG data)
                eeg_state = self.eeg_simulator.get_current_state()
                
                # Adapt based on EEG and ambient analysis
                if len(self.noise_buffer) > 10:
                    self._adapt_to_user_state(eeg_state)
                
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Adaptation error: {e}")
            
            time.sleep(5)  # Adapt every 5 seconds
    
    def _determine_optimal_noise_type(self):
        """Determine the best noise type based on current conditions"""
        if not self.noise_buffer:
            return self.user_profile['noise_preferences'].get(self.current_mode, 'pink')
        
        recent_analysis = list(self.noise_buffer)[-10:]  # Last 10 samples
        avg_noise_level = np.mean([a['rms_level'] for a in recent_analysis])
        avg_spectral_centroid = np.mean([a['spectral_centroid'] for a in recent_analysis])
        
        # Adaptive noise type selection
        if self.current_mode == "focus":
            if avg_noise_level > 0.1:  # High ambient noise
                return 'pink'  # Pink noise masks well
            elif avg_spectral_centroid > 2000:  # High frequency noise
                return 'brown'  # Brown noise for masking
            else:
                return 'binaural'  # Binaural for focus enhancement
        
        elif self.current_mode == "sleep":
            if avg_noise_level > 0.05:
                return 'brown'  # Deep, consistent masking
            else:
                return 'pink'  # Gentle background
        
        else:  # adaptive mode
            return self._adaptive_noise_selection(recent_analysis)
    
    def _adaptive_noise_selection(self, recent_analysis):
        """AI-based adaptive noise selection"""
        # Simple clustering-based approach
        features = np.array([[a['rms_level'], a['spectral_centroid'], a['zero_crossing_rate']] 
                           for a in recent_analysis])
        
        if len(features) > 3:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # Select noise type based on cluster characteristics
            if np.mean(features[clusters == 0, 0]) > np.mean(features[clusters == 1, 0]):
                return 'brown'  # Higher noise environment
            else:
                return 'pink'   # Lower noise environment
        
        return 'pink'  # Default
    
    def _calculate_noise_parameters(self):
        """Calculate optimal noise parameters"""
        if not self.noise_buffer:
            return {'volume': 0.3, 'frequency_range': [20, 20000]}
        
        recent_analysis = list(self.noise_buffer)[-5:]
        avg_noise_level = np.mean([a['rms_level'] for a in recent_analysis])
        
        # Adaptive volume based on ambient noise
        target_volume = min(0.8, max(0.1, avg_noise_level * 2))
        
        return {
            'volume': target_volume,
            'frequency_range': [20, 8000],
            'modulation_depth': 0.1
        }
    
    def _generate_adaptive_noise(self, noise_type, params, phase):
        """Generate adaptive noise audio"""
        t = np.linspace(phase, phase + self.chunk_size / self.sample_rate, 
                       self.chunk_size, endpoint=False)
        
        # Generate base noise
        if noise_type in self.noise_types:
            mono_audio = self.noise_types[noise_type](t, params)
        else:
            mono_audio = self._generate_pink_noise(t, params)
        
        # Apply volume
        mono_audio *= params['volume']
        
        # Create stereo output
        stereo_audio = np.column_stack([mono_audio, mono_audio])
        
        return stereo_audio.astype(np.float32)
    
    def _generate_white_noise(self, t, params):
        """Generate white noise"""
        return np.random.normal(0, 0.1, len(t))
    
    def _generate_pink_noise(self, t, params):
        """Generate pink noise (1/f noise)"""
        # Simple pink noise approximation
        white = np.random.normal(0, 1, len(t))
        freqs = fft.fftfreq(len(white), 1/self.sample_rate)
        freqs[0] = 1  # Avoid division by zero
        
        # Apply 1/f characteristic
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        pink_filter[0] = 0
        
        white_fft = fft.fft(white)
        pink_fft = white_fft * pink_filter
        pink = np.real(fft.ifft(pink_fft))
        
        return pink * 0.1
    
    def _generate_brown_noise(self, t, params):
        """Generate brown noise (1/f² noise)"""
        white = np.random.normal(0, 1, len(t))
        # Simple integration to create brown noise
        brown = np.cumsum(white)
        brown = brown - np.mean(brown)  # Remove DC component
        return brown * 0.05
    
    def _generate_nature_sounds(self, t, params):
        """Generate nature-like sounds"""
        # Combination of filtered noise and gentle oscillations
        base_noise = self._generate_pink_noise(t, params)
        
        # Add gentle wind-like modulation
        wind = 0.3 * np.sin(2 * np.pi * 0.1 * t) * base_noise
        
        # Add occasional "bird chirp" like sounds
        chirp_prob = 0.001
        if np.random.random() < chirp_prob:
            chirp_freq = np.random.uniform(1000, 3000)
            chirp = 0.1 * np.sin(2 * np.pi * chirp_freq * t) * np.exp(-10 * t)
            wind += chirp
        
        return wind
    
    def _generate_binaural_beats(self, t, params):
        """Generate binaural beats for brainwave entrainment"""
        base_freq = 200  # Base frequency
        beat_freq = 10   # Beat frequency for focus (alpha waves)
        
        if self.current_mode == "sleep":
            beat_freq = 2  # Delta waves for sleep
        
        # This would typically output different frequencies to each ear
        # For mono output, we'll create a beating effect
        carrier = np.sin(2 * np.pi * base_freq * t)
        modulation = np.sin(2 * np.pi * beat_freq * t)
        
        return 0.2 * carrier * (1 + 0.3 * modulation)
    
    def _adapt_to_user_state(self, eeg_state):
        """Adapt noise generation based on user's current state"""
        if eeg_state['focus_level'] < 0.3 and self.current_mode == "focus":
            # User losing focus, increase stimulation
            self.user_profile['preferred_frequencies']['focus'][1] += 5
        elif eeg_state['relaxation_level'] < 0.3 and self.current_mode == "sleep":
            # User not relaxed enough, adjust accordingly
            pass
        
        # Save updated profile
        self._save_user_profile(self.user_profile)
    
    def _print_analysis_summary(self):
        """Print current analysis summary"""
        if not self.noise_buffer:
            return
        
        recent = list(self.noise_buffer)[-1]
        print(f"\r🎵 Ambient: {recent['rms_level']:.3f} | "
              f"Freq: {recent['spectral_centroid']:.0f}Hz | "
              f"Env: {recent['environment_type']} | "
              f"Mode: {self.current_mode}", end="")


class AmbientSoundClassifier:
    """Simple ambient sound classifier"""
    
    def __init__(self):
        self.environment_types = ['quiet', 'office', 'traffic', 'nature', 'music']
    
    def classify(self, audio_data):
        """Classify the ambient environment"""
        rms = np.sqrt(np.mean(audio_data**2))
        zcr = np.sum(np.diff(np.sign(audio_data)) != 0) / (2 * len(audio_data))
        
        # Simple rule-based classification
        if rms < 0.01:
            return 'quiet'
        elif rms > 0.1 and zcr > 0.1:
            return 'traffic'
        elif rms > 0.05 and zcr < 0.05:
            return 'music'
        elif 0.02 < rms < 0.08:
            return 'office'
        else:
            return 'nature'


class EEGSimulator:
    """Simulates EEG data for testing (replace with real EEG interface)"""
    
    def __init__(self):
        self.time_start = time.time()
    
    def get_current_state(self):
        """Simulate current brainwave state"""
        t = time.time() - self.time_start
        
        # Simulate changing focus and relaxation levels
        focus_level = 0.5 + 0.3 * np.sin(0.1 * t) + 0.1 * np.random.random()
        relaxation_level = 0.6 + 0.2 * np.cos(0.05 * t) + 0.1 * np.random.random()
        
        return {
            'focus_level': np.clip(focus_level, 0, 1),
            'relaxation_level': np.clip(relaxation_level, 0, 1),
            'alpha_power': np.random.uniform(0.3, 0.8),
            'beta_power': np.random.uniform(0.2, 0.7),
            'theta_power': np.random.uniform(0.1, 0.5),
            'delta_power': np.random.uniform(0.1, 0.4)
        }


def main():
    """Main function to demonstrate the Noise Optimizer"""
    print("🎧 Personalized Noise Optimizer")
    print("=" * 50)
    
    optimizer = NoiseOptimizer()
    
    print("\nAvailable modes:")
    print("1. Focus mode - Optimized for concentration")
    print("2. Sleep mode - Optimized for relaxation")  
    print("3. Adaptive mode - AI-powered adaptation")
    
    try:
        mode_choice = input("\nSelect mode (1/2/3) or press Enter for adaptive: ").strip()
        
        mode_map = {'1': 'focus', '2': 'sleep', '3': 'adaptive', '': 'adaptive'}
        selected_mode = mode_map.get(mode_choice, 'adaptive')
        
        print(f"\n🚀 Starting in {selected_mode} mode...")
        optimizer.start_optimization(selected_mode)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have the required dependencies installed:")
        print("pip install numpy scipy pyaudio librosa scikit-learn matplotlib")


if __name__ == "__main__":
    main()
