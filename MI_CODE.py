import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops #label-connects region | regionprops-extracts properties like area, perimeter
from skimage.morphology import dilation, disk, erosion #dilation → grow regions | erosion → shrink regions |disk → structuring element
from skimage.segmentation import find_boundaries #detects edges between regions
from PIL import Image #GIF saving
from scipy.interpolate import interp1d

class RealisticKoopmanMicrostructurePredictor:
    def __init__(self, image_path="000006.jpg"):
        self.image_path = image_path
        self.gray = None
        self.labels = None
        self.regions = None
        self.K = None
        self.lifted_states = []
        self.state_dim = 10
        self.initial_intensity = None

    def load_and_preprocess(self):
        img = cv2.imread(self.image_path)
        if img is None:
            print("[INFO] Image not found. Generating sample image...")
            self.gray = np.zeros((480, 480), dtype=np.uint8)
            cv2.putText(self.gray, "Sample", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        else:
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #greyscaling

        # Store initial intensity for proper reconstruction
        self.initial_intensity = np.mean(self.gray[self.gray > 0])

        self.gray = cv2.equalizeHist(self.gray) #histogram contrast(improves contrasts range)
        blur = cv2.GaussianBlur(self.gray, (5, 5), 0) #uses gaussian blur to remove noise
        thresh = cv2.adaptiveThreshold( #convert image to binary for each region
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) #to perform advanced noise remove on images,
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        self.labels = label(clean)
        self.regions = regionprops(self.labels) #each grain one labelled region

    def extract_features(self, label_map):  #converting image to numerical features
        props = regionprops(label_map)

        if len(props) == 0:
            return {
                'count': 1,
                'avg_area': 1,
                'total_boundary': 1,
                'compactness': 0.5,
                'aspect_ratio': 1,
                'total_area': 1
            }

        areas = np.array([r.area for r in props]) #area is grain size
        perimeters = np.array([r.perimeter for r in props]) #perimeter is boundary energy
        compactness = 4 * np.pi * areas / (perimeters**2 + 1e-6) #handling circular grains
        aspect_ratios = np.array([r.major_axis_length / (r.minor_axis_length + 1e-6) for r in props]) #used for shape elongation

        return {
            'count': len(props),
            'avg_area': np.mean(areas),
            'total_boundary': np.sum(perimeters),
            'compactness': np.mean(compactness),
            'aspect_ratio': np.mean(aspect_ratios),
            'total_area': np.sum(areas)
        }

    def get_lifted_state(self, features): #converting raw data into higher dimension
        c = np.clip(features['count'], 1, 1e6) #np.clip means preventing extreme values
        s = np.clip(features['avg_area'], 1, 1e6)
        b = np.clip(features['total_boundary'], 1, 1e6)
        comp = np.clip(features['compactness'], 1e-3, 1)
        ar = np.clip(features['aspect_ratio'], 0.1, 100)
        ta = np.clip(features['total_area'], 1, 1e6)
        lifted = np.array([
            c,                              # 0: Particle count (decreases with coarsening)
            s,                              # 1: Average grain size (increases with coarsening)
            b,                              # 2: Total boundary length (decreases)
            comp,                           # 3: Compactness (stays relatively stable)
            ar,                             # 4: Aspect ratio (stays relatively stable)
            ta,                             # 5: Total preserved area (stays constant)
            np.log(c + 1),                  # 6: Log of count (smooths exponential decay)
            np.sqrt(s),                     # 7: Sqrt of size (smooths growth)
            b / (c + 1e-6),                 # 8: Boundary per particle (decreases)
            s**1.5 / 1e6,                   # 9: Size growth indicator (smoothed)
        ], dtype=np.float64)
#state vector is constructed
        return lifted

    def build_koopman_operator(self, n_historical=6):
        print(f"\n[INFO] Building Koopman operator with {n_historical} physically-grounded states...")
        features_0 = self.extract_features(self.labels)
        state_0 = self.get_lifted_state(features_0)
        self.lifted_states = [state_0]

        states = [state_0]

        # Generate physically realistic temporal evolution
        # Based on grain growth kinetics: dD/dt ∝ D^n where n=2-4 for different mechanisms
        time_points = np.array([0, 1, 2, 3, 4, 5], dtype=float)[:n_historical]

        for step, t in enumerate(time_points[1:], 1):
            # Coarsening kinetics: particles merge, grains grow
            growth_exponent = 2.5  # Typical for grain growth
            decay_rate = 0.92  # Particles reduce per step

            size_growth = (1 + 0.15 * t) ** (1 / growth_exponent)
            count_decay = decay_rate ** step
            total_area_preserved = features_0['total_area']

            features_new = {
                'count': max(1, features_0['count'] * count_decay),
                'avg_area': features_0['avg_area'] * size_growth,
                'total_boundary': features_0['total_boundary'] * (count_decay ** 0.8),  # Boundary decreases slower
                'compactness': features_0['compactness'] * (0.995 ** step),  # Slight increase in regularity
                'aspect_ratio': features_0['aspect_ratio'] * (0.998 ** step),  # Becomes more equiaxed
                'total_area': total_area_preserved  # CONSTRAINT: Total area conserved
            }

            state = self.get_lifted_state(features_new)
            states.append(state)
            self.lifted_states.append(state)

        # Build Koopman operator with proper dimensions
        X = np.array(states[:-1]).T  # (state_dim, n_historical-1)
        X_next = np.array(states[1:]).T  # (state_dim, n_historical-1)

        print(f"[DEBUG] X shape: {X.shape}, X_next shape: {X_next.shape}")

        try:
            X_pinv = np.linalg.pinv(X, rcond=1e-4)
            self.K = X_next @ X_pinv

            print(f"[DEBUG] Koopman operator K shape: {self.K.shape}")
            u, s, vt = np.linalg.svd(self.K) #used SVD for checking stability
            print(f"[INFO] Singular values: {s}")

        except Exception as e:
            print(f"[ERROR] Failed to build Koopman operator: {e}")
            raise

        return self
    def predict_future_states(self, n_steps):
        if self.K is None:
            raise ValueError("Koopman operator not built.")

        print(f"\n[INFO] Predicting {n_steps} future states with physical constraints...")

        current_state = self.lifted_states[-1].copy().reshape(-1, 1)
        predictions = []

        initial_area = self.lifted_states[0][5]  # Total area

        for step in range(n_steps):
            next_state = (self.K @ current_state).flatten()
            prev_size = current_state.flatten()[1]
            next_state[1] = np.maximum(next_state[1], prev_size * 1.02)  # At least 2% growth

            prev_count = current_state.flatten()[0]
            next_state[0] = np.minimum(next_state[0], prev_count * 0.95)  # At most 5% decrease
            next_state[0] = np.maximum(next_state[0], 1)  # At least 1 particle

            # 3. Total area conservation
            next_state[5] = initial_area

            # 4. Boundary decreases with particle count
            next_state[2] = np.minimum(next_state[2], current_state.flatten()[2] * 0.98)

            # 5. Update derived terms based on constraints
            next_state[6] = np.log(next_state[0] + 1)
            next_state[7] = np.sqrt(next_state[1])
            next_state[8] = next_state[2] / (next_state[0] + 1e-6)
            next_state[9] = (next_state[1]**1.5) / 1e6

            next_state = np.clip(next_state, 1e-3, 1e6)

            predictions.append(next_state)
            current_state = next_state.reshape(-1, 1)

            print(f"  Step {step+1:2d}: Count={next_state[0]:7.1f}, AvgSize={next_state[1]:7.2f}, "
                  f"Boundary={next_state[2]:9.1f}, TotalArea={next_state[5]:8.1f}")

        return predictions

    def lift_to_image(self, lifted_state, original_labels, step_num): #converting predicted state to image
        current_features = self.extract_features(original_labels)
        predicted_size = lifted_state[1]

        if current_features['avg_area'] > 0:
            growth_ratio = predicted_size / current_features['avg_area']
        else:
            growth_ratio = 1

        # Controlled dilation: grow grains but not too aggressively
        dilation_factor = int(np.clip(1 + 0.3 * growth_ratio, 1, 3))

        # Apply dilation to grow grains
        dilated = dilation(original_labels, disk(dilation_factor))

        return dilated, dilation_factor

    def reconstruct(self, original_gray, pred_labels, step_num):
        recon = np.zeros_like(original_gray, dtype=np.float32)
        unique_labels = np.unique(pred_labels)

        # Extract original brightness
        original_intensities = []
        for lab in np.unique(self.labels):
            if lab == 0:
                continue
            mask = self.labels == lab
            if mask.sum() > 0:
                original_intensities.append(np.mean(original_gray[mask]))

        # Use realistic intensity distribution
        intensity_mean = np.mean(original_intensities) if original_intensities else 128
        intensity_std = np.std(original_intensities) if original_intensities else 15

        # Assign intensities to new grains
        for lab in unique_labels:
            if lab == 0:
                continue
            mask = pred_labels == lab

            # Assign intensity close to original mean, with small variation
            grain_intensity = np.random.normal(intensity_mean, intensity_std * 0.5)
            grain_intensity = np.clip(grain_intensity, 50, 230)

            recon[mask] = grain_intensity

        # Enhance boundaries (phase boundaries appear darker)
        boundaries = find_boundaries(pred_labels)
        recon[boundaries] = np.clip(recon[boundaries] * 0.85, 0, 255)

        # Add minimal realistic noise
        recon += np.random.normal(0, 1.0, recon.shape)

        return np.clip(recon, 0, 255).astype(np.uint8)

    def generate_prediction_gif(self, n_steps, output_path="microstructure_evolution.gif", fps=2):
#Build Koopman
#Predict states
#Convert each image
#Save as GIF
        print(f"\n[INFO] Generating {n_steps}-step realistic prediction GIF...")

        self.build_koopman_operator(n_historical=6)
        predictions = self.predict_future_states(n_steps)

        frames = []
        frame_data = []

        # Original frame
        original_frame_rgb = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2RGB)
        frames.append(Image.fromarray(cv2.cvtColor(original_frame_rgb, cv2.COLOR_BGR2RGB)))
        frame_data.append(("ORIGINAL", self.gray, 0))

        # Predicted frames
        current_labels = self.labels.copy()

        for step, lifted_state in enumerate(predictions):
            try:
                pred_labels, dilation_factor = self.lift_to_image(lifted_state, current_labels, step + 1)
                reconstructed = self.reconstruct(self.gray, pred_labels, step + 1)

                frame_rgb = cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2RGB)
                frames.append(Image.fromarray(frame_rgb))
                frame_data.append((f"Step {step+1}", reconstructed, dilation_factor))

                current_labels = pred_labels

                print(f"  Frame {step+1}/{n_steps} generated")

            except Exception as e:
                print(f"[WARNING] Error generating frame {step+1}: {e}")
                continue

        # Save GIF
        if len(frames) > 1:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0,
                optimize=False
            )
            print(f"\n[SUCCESS] GIF saved to: {output_path}")

        return frames, frame_data

    def visualize_prediction(self, frame_data):
        """Visualize all prediction frames."""
        n_frames = len(frame_data)
        cols = 4
        rows = (n_frames + cols - 1) // cols

        plt.figure(figsize=(20, 5 * rows))

        for idx, (title, img, dilate_factor) in enumerate(frame_data):
            plt.subplot(rows, cols, idx + 1)
            plt.title(f"{title}\n(Dilation: {dilate_factor})", fontsize=10, fontweight='bold')
            plt.imshow(img, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("="*70)
    print("REALISTIC KOOPMAN OPERATOR-BASED MICROSTRUCTURE PREDICTION")
    print("="*70)

    predictor = RealisticKoopmanMicrostructurePredictor(image_path="000006.jpg")
    predictor.load_and_preprocess()

    initial_features = predictor.extract_features(predictor.labels)
    print("\n" + "="*70)
    print("INITIAL MICROSTRUCTURE FEATURES")
    print("="*70)
    for key, value in initial_features.items():
        print(f"  {key.upper():<20}: {value:.4f}")

    print("\n" + "="*70)
    while True:
        try:
            n_future_states = int(input("Enter number of future states to predict (1-20): "))
            if 1 <= n_future_states <= 20:
                break
        except ValueError:
            pass

    print("="*70)
    try:
        frames, frame_data = predictor.generate_prediction_gif(
            n_steps=n_future_states,
            output_path=f"microstructure_evolution_{n_future_states}steps.gif",
            fps=2
        )

        print("\n[INFO] Displaying prediction visualization...")
        predictor.visualize_prediction(frame_data)

        print("\n" + "="*70)
        print(f"✓ PREDICTION COMPLETE: {n_future_states} future states generated")
        print(f"✓ GIF saved as: microstructure_evolution_{n_future_states}steps.gif")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()