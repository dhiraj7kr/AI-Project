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
        self.current_mode = "focus"  # Will be set by user selection
        self.available_modes = {
            'focus': 'Deep Focus - Binaural beats for concentration',
            'sleep': 'Deep Sleep - Low frequency for rest',
            'study': 'Study Mode - Pink noise for learning',
            'meditation': 'Meditation - Tibetan bowls & nature',
            'creative': 'Creative Flow - Dynamic ambient sounds',
            'anxiety': 'Anxiety Relief - Calming frequencies',
            'workout': 'Workout - Energizing rhythmic noise',
            'reading': 'Reading - Minimal distraction sounds',
            'coding': 'Coding - Consistent background hum',
            'meeting': 'Meeting - Subtle masking noise',
            'nap': 'Power Nap - Quick rest optimization',
            'storm': 'Storm Sounds - Rain and thunder',
            'cafe': 'Cafe Ambience - Social background noise',
            'forest': 'Forest Sounds - Deep nature immersion',
            'adaptive': 'AI Adaptive - Smart mode switching'
        }
        
        self.noise_types = {
            'white': self._generate_white_noise,
            'pink': self._generate_pink_noise,
            'brown': self._generate_brown_noise,
            'nature': self._generate_nature_sounds,
            'binaural': self._generate_binaural_beats,
            'tibetan': self._generate_tibetan_bowls,
            'rain': self._generate_rain_sounds,
            'thunder': self._generate_thunder_sounds,
            'ocean': self._generate_ocean_waves,
            'fire': self._generate_fire_crackling,
            'wind': self._generate_wind_sounds,
            'cafe_noise': self._generate_cafe_ambience,
            'rhythmic': self._generate_rhythmic_noise,
            'harmonic': self._generate_harmonic_series,
            'pulse': self._generate_pulse_waves
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
        """Determine the best noise type based on current conditions and mode"""
        if not self.noise_buffer:
            return self._get_default_noise_type()
        
        recent_analysis = list(self.noise_buffer)[-10:]  # Last 10 samples
        avg_noise_level = np.mean([a['rms_level'] for a in recent_analysis])
        avg_spectral_centroid = np.mean([a['spectral_centroid'] for a in recent_analysis])
        environment = recent_analysis[-1]['environment_type']
        
        # Mode-specific noise selection with environmental adaptation
        mode_strategies = {
            'focus': self._focus_mode_selection,
            'sleep': self._sleep_mode_selection,
            'study': self._study_mode_selection,
            'meditation': self._meditation_mode_selection,
            'creative': self._creative_mode_selection,
            'anxiety': self._anxiety_mode_selection,
            'workout': self._workout_mode_selection,
            'reading': self._reading_mode_selection,
            'coding': self._coding_mode_selection,
            'meeting': self._meeting_mode_selection,
            'nap': self._nap_mode_selection,
            'storm': self._storm_mode_selection,
            'cafe': self._cafe_mode_selection,
            'forest': self._forest_mode_selection,
            'adaptive': self._adaptive_noise_selection
        }
        
        strategy = mode_strategies.get(self.current_mode, self._adaptive_noise_selection)
        return strategy(avg_noise_level, avg_spectral_centroid, environment)
    
    def _get_default_noise_type(self):
        """Get default noise type for each mode"""
        defaults = {
            'focus': 'binaural',
            'sleep': 'brown',
            'study': 'pink',
            'meditation': 'tibetan',
            'creative': 'nature',
            'anxiety': 'ocean',
            'workout': 'rhythmic',
            'reading': 'pink',
            'coding': 'brown',
            'meeting': 'white',
            'nap': 'rain',
            'storm': 'thunder',
            'cafe': 'cafe_noise',
            'forest': 'nature',
            'adaptive': 'pink'
        }
        return defaults.get(self.current_mode, 'pink')
    
    def _focus_mode_selection(self, noise_level, spectral_centroid, environment):
        """Focus mode noise selection strategy"""
        if noise_level > 0.1:  # High ambient noise
            return 'brown'  # Strong masking
        elif environment == 'office':
            return 'binaural'  # Binaural beats for focus
        elif spectral_centroid > 2000:
            return 'pink'  # Counter high frequencies
        else:
            return 'harmonic'  # Harmonic series for deep focus
    
    def _sleep_mode_selection(self, noise_level, spectral_centroid, environment):
        """Sleep mode noise selection strategy"""
        if noise_level > 0.08:
            return 'brown'  # Strong masking for noisy environments
        elif environment == 'traffic':
            return 'ocean'  # Natural masking
        else:
            return 'rain'  # Gentle rain for sleep
    
    def _study_mode_selection(self, noise_level, spectral_centroid, environment):
        """Study mode noise selection strategy"""
        if environment == 'music':
            return 'white'  # Mask musical distractions
        elif noise_level > 0.06:
            return 'pink'  # Good for concentration with some masking
        else:
            return 'brown'  # Deep, consistent background
    
    def _meditation_mode_selection(self, noise_level, spectral_centroid, environment):
        """Meditation mode noise selection strategy"""
        if environment == 'traffic' or noise_level > 0.08:
            return 'ocean'  # Natural masking
        elif environment == 'quiet':
            return 'tibetan'  # Tibetan bowls for meditation
        else:
            return 'wind'  # Gentle wind sounds
    
    def _creative_mode_selection(self, noise_level, spectral_centroid, environment):
        """Creative mode noise selection strategy"""
        if environment == 'office':
            return 'nature'  # Inspiring natural sounds
        elif noise_level < 0.03:
            return 'fire'  # Crackling fire for creativity
        else:
            return 'ocean'  # Dynamic ocean sounds
    
    def _anxiety_mode_selection(self, noise_level, spectral_centroid, environment):
        """Anxiety relief mode noise selection strategy"""
        if spectral_centroid > 1500:  # High frequency anxiety triggers
            return 'brown'  # Deep, calming brown noise
        elif environment == 'traffic':
            return 'ocean'  # Calming ocean waves
        else:
            return 'rain'  # Gentle rain sounds
    
    def _workout_mode_selection(self, noise_level, spectral_centroid, environment):
        """Workout mode noise selection strategy"""
        return 'rhythmic'  # Always use rhythmic noise for workouts
    
    def _reading_mode_selection(self, noise_level, spectral_centroid, environment):
        """Reading mode noise selection strategy"""
        if environment == 'music' or spectral_centroid > 2000:
            return 'brown'  # Mask distracting sounds
        elif noise_level > 0.05:
            return 'pink'  # Good masking without distraction
        else:
            return 'white'  # Minimal, consistent background
    
    def _coding_mode_selection(self, noise_level, spectral_centroid, environment):
        """Coding mode noise selection strategy"""
        if environment == 'office':
            return 'brown'  # Consistent masking
        elif noise_level > 0.07:
            return 'pink'  # Good masking
        else:
            return 'harmonic'  # Structured harmonic series
    
    def _meeting_mode_selection(self, noise_level, spectral_centroid, environment):
        """Meeting mode noise selection strategy"""
        return 'white'  # Subtle, non-distracting masking
    
    def _nap_mode_selection(self, noise_level, spectral_centroid, environment):
        """Power nap mode noise selection strategy"""
        if noise_level > 0.06:
            return 'brown'  # Strong masking for quick sleep
        else:
            return 'rain'  # Gentle rain for napping
    
    def _storm_mode_selection(self, noise_level, spectral_centroid, environment):
        """Storm sounds mode selection strategy"""
        if noise_level > 0.05:
            return 'thunder'  # Full storm experience
        else:
            return 'rain'  # Rain component of storm
    
    def _cafe_mode_selection(self, noise_level, spectral_centroid, environment):
        """Cafe ambience mode selection strategy"""
        return 'cafe_noise'  # Always use cafe sounds
    
    def _forest_mode_selection(self, noise_level, spectral_centroid, environment):
        """Forest sounds mode selection strategy"""
        if environment == 'traffic':
            return 'wind'  # Wind through trees
        else:
            return 'nature'  # Full forest experience
    
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
        """Calculate optimal noise parameters based on mode and environment"""
        if not self.noise_buffer:
            return self._get_default_parameters()
        
        recent_analysis = list(self.noise_buffer)[-5:]
        avg_noise_level = np.mean([a['rms_level'] for a in recent_analysis])
        
        # Mode-specific parameter calculation
        mode_params = {
            'focus': {'base_volume': 0.4, 'freq_range': [40, 8000], 'modulation': 0.1},
            'sleep': {'base_volume': 0.2, 'freq_range': [20, 500], 'modulation': 0.05},
            'study': {'base_volume': 0.3, 'freq_range': [50, 4000], 'modulation': 0.08},
            'meditation': {'base_volume': 0.25, 'freq_range': [30, 2000], 'modulation': 0.15},
            'creative': {'base_volume': 0.35, 'freq_range': [40, 6000], 'modulation': 0.2},
            'anxiety': {'base_volume': 0.3, 'freq_range': [20, 1000], 'modulation': 0.05},
            'workout': {'base_volume': 0.6, 'freq_range': [60, 8000], 'modulation': 0.3},
            'reading': {'base_volume': 0.25, 'freq_range': [50, 3000], 'modulation': 0.05},
            'coding': {'base_volume': 0.35, 'freq_range': [40, 4000], 'modulation': 0.1},
            'meeting': {'base_volume': 0.15, 'freq_range': [100, 8000], 'modulation': 0.02},
            'nap': {'base_volume': 0.25, 'freq_range': [20, 800], 'modulation': 0.05},
            'storm': {'base_volume': 0.5, 'freq_range': [20, 10000], 'modulation': 0.4},
            'cafe': {'base_volume': 0.4, 'freq_range': [100, 6000], 'modulation': 0.25},
            'forest': {'base_volume': 0.3, 'freq_range': [30, 5000], 'modulation': 0.3},
            'adaptive': {'base_volume': 0.3, 'freq_range': [20, 8000], 'modulation': 0.1}
        }
        
        base_params = mode_params.get(self.current_mode, mode_params['adaptive'])
        
        # Adaptive volume based on ambient noise
        volume_multiplier = min(2.0, max(0.5, 1 + avg_noise_level * 3))
        target_volume = min(0.8, base_params['base_volume'] * volume_multiplier)
        
        return {
            'volume': target_volume,
            'frequency_range': base_params['freq_range'],
            'modulation_depth': base_params['modulation'],
            'mode': self.current_mode
        }
    
    def _get_default_parameters(self):
        """Get default parameters when no analysis is available"""
        return {
            'volume': 0.3,
            'frequency_range': [20, 8000],
            'modulation_depth': 0.1,
            'mode': self.current_mode
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
        base_freq = 200
        
        # Mode-specific binaural frequencies
        beat_frequencies = {
            'focus': 40,      # Gamma waves for focus
            'study': 15,      # Beta waves for learning
            'meditation': 8,  # Alpha waves for meditation
            'creative': 10,   # Alpha waves for creativity  
            'anxiety': 6,     # Theta waves for calm
            'reading': 12,    # SMR waves for attention
            'coding': 16,     # Beta waves for logic
            'sleep': 2,       # Delta waves for sleep
            'nap': 4          # Theta waves for light sleep
        }
        
        beat_freq = beat_frequencies.get(self.current_mode, 10)
        
        # Generate beating effect
        carrier = np.sin(2 * np.pi * base_freq * t)
        modulation = np.sin(2 * np.pi * beat_freq * t)
        
        return 0.2 * carrier * (1 + 0.3 * modulation)
    
    def _generate_tibetan_bowls(self, t, params):
        """Generate Tibetan singing bowl sounds"""
        # Multiple harmonic frequencies typical of singing bowls
        frequencies = [256, 341, 427, 512, 682]  # Hz
        sound = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            amplitude = 0.3 / (i + 1)  # Decreasing amplitude for harmonics
            decay = np.exp(-0.5 * t)   # Natural decay
            harmonic = amplitude * np.sin(2 * np.pi * freq * t) * decay
            sound += harmonic
        
        # Add gentle modulation
        modulation = 1 + 0.1 * np.sin(2 * np.pi * 0.2 * t)
        return sound * modulation * 0.4
    
    def _generate_rain_sounds(self, t, params):
        """Generate realistic rain sounds"""
        # Base rain noise (filtered white noise)
        rain_base = np.random.normal(0, 0.1, len(t))
        
        # Apply high-pass filter for rain characteristic
        b, a = scipy.signal.butter(4, 1000, btype='high', fs=self.sample_rate)
        rain_filtered = scipy.signal.filtfilt(b, a, rain_base)
        
        # Add occasional droplet sounds
        droplet_prob = 0.002
        for i in range(len(t)):
            if np.random.random() < droplet_prob:
                droplet_freq = np.random.uniform(2000, 8000)
                droplet_decay = np.exp(-50 * (t[i:] - t[i] if i < len(t) else 0))
                if len(droplet_decay) > 0:
                    droplet = 0.05 * np.sin(2 * np.pi * droplet_freq * (t[i:] - t[i]))
                    rain_filtered[i:i+len(droplet_decay)] += droplet[:len(rain_filtered[i:])] * droplet_decay[:len(rain_filtered[i:])]
        
        return rain_filtered * 0.6
    
    def _generate_thunder_sounds(self, t, params):
        """Generate thunder and storm sounds"""
        # Base rain
        rain = self._generate_rain_sounds(t, params) * 0.7
        
        # Occasional thunder
        thunder_prob = 0.0005  # Very occasional
        thunder = np.zeros_like(t)
        
        if np.random.random() < thunder_prob:
            # Thunder rumble (low frequency noise burst)
            thunder_start = int(len(t) * np.random.random())
            thunder_duration = int(self.sample_rate * 2)  # 2 second thunder
            
            if thunder_start + thunder_duration < len(t):
                rumble = np.random.normal(0, 1, thunder_duration)
                # Low-pass filter for rumble effect
                b, a = scipy.signal.butter(4, 200, btype='low', fs=self.sample_rate)
                rumble = scipy.signal.filtfilt(b, a, rumble)
                
                # Envelope for natural thunder shape
                envelope = np.exp(-np.linspace(0, 3, thunder_duration))
                thunder[thunder_start:thunder_start+thunder_duration] = rumble * envelope * 0.3
        
        return rain + thunder
    
    def _generate_ocean_waves(self, t, params):
        """Generate ocean wave sounds"""
        # Base ocean noise (filtered pink noise)
        ocean_base = self._generate_pink_noise(t, params)
        
        # Apply band-pass filter for ocean characteristics
        b, a = scipy.signal.butter(4, [50, 2000], btype='band', fs=self.sample_rate)
        ocean_filtered = scipy.signal.filtfilt(b, a, ocean_base)
        
        # Add wave rhythm (slow modulation)
        wave_freq = 0.1  # Very slow waves
        wave_modulation = 1 + 0.4 * np.sin(2 * np.pi * wave_freq * t)
        
        return ocean_filtered * wave_modulation * 0.5
    
    def _generate_fire_crackling(self, t, params):
        """Generate fire crackling sounds"""
        # Base crackling (burst noise)
        crackling = np.random.uniform(-0.1, 0.1, len(t))
        
        # Add random pops and crackles
        pop_prob = 0.001
        for i in range(len(t)):
            if np.random.random() < pop_prob:
                pop_freq = np.random.uniform(800, 4000)
                pop_duration = int(self.sample_rate * 0.1)  # 0.1 second pops
                
                if i + pop_duration < len(t):
                    pop_envelope = np.exp(-10 * np.linspace(0, 1, pop_duration))
                    pop_sound = 0.1 * np.sin(2 * np.pi * pop_freq * np.linspace(0, 0.1, pop_duration))
                    crackling[i:i+pop_duration] += pop_sound * pop_envelope
        
        return crackling * 0.4
    
    def _generate_wind_sounds(self, t, params):
        """Generate wind sounds"""
        # Base wind (filtered brown noise)
        wind_base = self._generate_brown_noise(t, params)
        
        # Apply band-pass filter for wind characteristics
        b, a = scipy.signal.butter(4, [100, 1500], btype='band', fs=self.sample_rate)
        wind_filtered = scipy.signal.filtfilt(b, a, wind_base)
        
        # Add wind gusts (slow modulation)
        gust_freq = 0.05
        gust_modulation = 1 + 0.6 * np.sin(2 * np.pi * gust_freq * t)
        
        return wind_filtered * gust_modulation * 0.3
    
    def _generate_cafe_ambience(self, t, params):
        """Generate cafe ambience sounds"""
        # Base chatter (filtered noise)
        chatter = np.random.normal(0, 0.05, len(t))
        
        # Apply band-pass filter for human voice range
        b, a = scipy.signal.butter(4, [200, 3000], btype='band', fs=self.sample_rate)
        chatter_filtered = scipy.signal.filtfilt(b, a, chatter)
        
        # Add occasional cup clinks and movements
        clink_prob = 0.0008
        for i in range(len(t)):
            if np.random.random() < clink_prob:
                clink_freq = np.random.uniform(1000, 3000)
                clink_duration = int(self.sample_rate * 0.05)
                
                if i + clink_duration < len(t):
                    clink_envelope = np.exp(-20 * np.linspace(0, 1, clink_duration))
                    clink_sound = 0.02 * np.sin(2 * np.pi * clink_freq * np.linspace(0, 0.05, clink_duration))
                    chatter_filtered[i:i+clink_duration] += clink_sound * clink_envelope
        
        return chatter_filtered * 0.6
    
    def _generate_rhythmic_noise(self, t, params):
        """Generate rhythmic noise for workouts"""
        # Base rhythmic pulse
        pulse_freq = 2.0  # 120 BPM / 60 = 2 Hz
        pulse = np.sin(2 * np.pi * pulse_freq * t)
        
        # Add noise component
        noise = self._generate_pink_noise(t, params) * 0.3
        
        # Combine with rhythm emphasis
        rhythmic = noise * (1 + 0.5 * pulse)
        
        # Add energizing higher frequencies
        energy = 0.2 * np.sin(2 * np.pi * 440 * t) * (pulse > 0)
        
        return rhythmic + energy
    
    def _generate_harmonic_series(self, t, params):
        """Generate harmonic series for focus"""
        base_freq = 110  # A2 note
        harmonics = np.zeros_like(t)
        
        # Generate first 8 harmonics
        for i in range(1, 9):
            amplitude = 0.5 / i  # Decreasing amplitude
            frequency = base_freq * i
            harmonics += amplitude * np.sin(2 * np.pi * frequency * t)
        
        return harmonics * 0.3
    
    def _generate_pulse_waves(self, t, params):
        """Generate pulse waves"""
        freq = 200
        duty_cycle = 0.3  # 30% duty cycle
        
        # Generate square wave with specified duty cycle
        pulse = scipy.signal.square(2 * np.pi * freq * t, duty=duty_cycle)
        
        # Soften the edges
        b, a = scipy.signal.butter(4, 1000, btype='low', fs=self.sample_rate)
        pulse_filtered = scipy.signal.filtfilt(b, a, pulse)
        
        return pulse_filtered * 0.2
    
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
    
    print("\n🎵 Available Modes:")
    print("=" * 30)
    
    modes = list(optimizer.available_modes.keys())
    for i, (mode, description) in enumerate(optimizer.available_modes.items(), 1):
        print(f"{i:2d}. {mode.upper():12s} - {description}")
    
    print("\n💡 Special Features:")
    print("   • Real-time ambient sound analysis")
    print("   • AI-powered noise adaptation")
    print("   • Binaural beats for brainwave entrainment")
    print("   • Environmental sound masking")
    print("   • Personalized learning system")
    
    try:
        mode_choice = input(f"\nSelect mode (1-{len(modes)}) or press Enter for adaptive: ").strip()
        
        if mode_choice == "":
            selected_mode = 'adaptive'
        else:
            try:
                mode_index = int(mode_choice) - 1
                if 0 <= mode_index < len(modes):
                    selected_mode = modes[mode_index]
                else:
                    print("Invalid selection, using adaptive mode")
                    selected_mode = 'adaptive'
            except ValueError:
                print("Invalid input, using adaptive mode")
                selected_mode = 'adaptive'
        
        print(f"\n🚀 Starting {selected_mode.upper()} mode...")
        print(f"📋 {optimizer.available_modes[selected_mode]}")
        print("\n🎛️  Real-time Analysis:")
        print("   Ambient | Frequency | Environment | Current Mode")
        print("-" * 50)
        
        optimizer.start_optimization(selected_mode)
        
    except KeyboardInterrupt:
        print("\n👋 Thanks for using Noise Optimizer!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Installation Requirements:")
        print("pip install numpy scipy pyaudio librosa scikit-learn matplotlib")
        print("\n🔧 Troubleshooting:")
        print("• Check microphone permissions")
        print("• Ensure audio drivers are updated")
        print("• Try running as administrator (Windows)")


if __name__ == "__main__":
    main()
    print("pip install numpy scipy pyaudio librosa scikit-learn matplotlib")


if __name__ == "__main__":
    main()
