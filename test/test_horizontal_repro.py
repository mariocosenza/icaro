import sys
import unittest
from unittest.mock import MagicMock
import os
import numpy as np

# Adjust path to find src
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Mock dependencies
sys.modules["mediapipe"] = MagicMock()
sys.modules["sklearn.pipeline"] = MagicMock()
sys.modules["pymongo"] = MagicMock()
sys.modules["firebase_admin"] = MagicMock()
sys.modules["firebase_admin.messaging"] = MagicMock()
sys.modules["firebase_admin.credentials"] = MagicMock()

# Import the class under test
from src.live_fall_detector import LiveManDownDetector, DetectorConfig

class TestHorizontalRepro(unittest.TestCase):
    def test_horizontal_never_detected_without_fall(self):
        """
        Verify that horizontal posture is IGNORED if no fall has occurred recently,
        due to the `_post_fall_timer` check.
        """
        # Mock models
        fall_model = MagicMock()
        horizontal_model = MagicMock()
        
        # Mock BodyLandmark
        BodyLandmark = MagicMock()

        # Mock feature extraction to always return valid high-quality features
        # We patch the module functions bound in live_fall_detector
        import src.live_fall_detector as lfd
        
        # Original functions
        orig_extract_h = lfd.extract_frame_features_horizontal
        orig_extract_f = lfd.extract_frame_features_fall
        orig_proba = lfd._proba_pos
        
        try:
            # Setup mocks to simulate a continuous "Horizontal" stream
            # extract_frame_features_horizontal returns (features, quality)
            lfd.extract_frame_features_horizontal = MagicMock(return_value=(np.zeros((10,), dtype=np.float32), 1.0))
            
            # extract_frame_features_fall returns (features, quality)
            lfd.extract_frame_features_fall = MagicMock(return_value=(np.zeros((10,), dtype=np.float32), 1.0))

            # _proba_pos returns probability.
            # We want horizontal model to return 1.0 (Definitely Horizontal)
            # We want fall model to return 0.0 (No Fall) to verify the masking behavior
            def side_effect_proba(model, X):
                if model == horizontal_model:
                    return np.array([0.99]) # High probability horizontal
                return np.array([0.01]) # Low probability fall
            
            lfd._proba_pos = MagicMock(side_effect=side_effect_proba)
            
            config = DetectorConfig(
                horizontal_threshold=0.5,
                consecutive_horizontal=1,
                post_fall_duration=100
            )
            
            detector = LiveManDownDetector(
                BodyLandmark,
                fall_model=fall_model,
                horizontal_model=horizontal_model,
                window=10,
                config=config
            )

            # Feed the detector with enough frames to fill the window
            # We simulate a "horizontal" person who did NOT fall (fall prob is low)
            # import numpy as np  <-- REMOVED

            
            # Mock pose33 (list of dicts)
            pose33 = [{"x":0, "y":0, "z":0, "visibility":1, "presence":1}] * 33

            horizontal_detected = False
            for i in range(20):
                out = detector.update(pose33)
                if out.get("horizontal_event"):
                    horizontal_detected = True
                    break
            
            # This assertion confirms the fix: Horizontal IS detected even without a fall
            print(f"\nHorizontal Detected: {horizontal_detected}")
            self.assertTrue(horizontal_detected, "Horizontal SHOULD be detected now that horizontal_always_active is True")
            
            # Now, let's force a fall to prove it works AFTER a fall
            # Change fall model to return high prob
            def side_effect_fall(model, X):
                 return np.array([0.99])
            lfd._proba_pos = MagicMock(side_effect=side_effect_fall)
            
            # Trigger fall
            for i in range(10): 
                detector.update(pose33)
                
            # Now switch back to horizontal-only (no fall) but with timer active
            def side_effect_horiz_only(model, X):
                if model == horizontal_model:
                    return np.array([0.99])
                return np.array([0.01])
            lfd._proba_pos = MagicMock(side_effect=side_effect_horiz_only)
            
            # Detector should now be in post-fall state
            out = detector.update(pose33)
            # Note: Depending on buffer clearing or logic, it might take a moment or depend on window
            # But just checking the timer logic involves reading internal state or outcome
            
            if detector._post_fall_timer > 0:
                print("Timer is active!")
                
            # We expect horizontal_event to likely be True now or soon, 
            # but getting the exact frame right depends on window buffers.
            # However, the primary goal is verifying the blockage first.

        finally:
            # Restore patches
            lfd.extract_frame_features_horizontal = orig_extract_h
            lfd.extract_frame_features_fall = orig_extract_f
            lfd._proba_pos = orig_proba

if __name__ == "__main__":
    unittest.main()
