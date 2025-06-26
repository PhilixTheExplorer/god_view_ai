# Multi-Angle CCTV Camera Pose Detection Improvements

## Overview

Enhanced the God View monitoring system to handle CCTV camera views from multiple angles (front, back, left, right, and diagonal views) for more accurate pose detection and fall detection.

## Key Changes Made

### 1. Enhanced PoseAnalyzer Class (`pose_analyzer.py`)

#### **Camera Angle Detection**

- Added `_estimate_viewing_angle()` method to automatically detect camera viewing angle
- Analyzes keypoint visibility patterns and shoulder positions
- Identifies 7 different viewing angles:
  - `front` - Frontal view with face visible
  - `back` - Back view with minimal face visibility
  - `left_side` - Left profile view
  - `right_side` - Right profile view
  - `diagonal_front` - Diagonal front view
  - `diagonal_back` - Diagonal back view
  - Default fallback to `front`

#### **View-Specific Pose Classification**

- `_classify_frontal_view()` - For front/back camera angles

  - Uses vertical alignment of head → shoulders → hips → ankles
  - Analyzes normal body progression for upright postures
  - Detects compressed alignment for sitting
  - Identifies horizontal orientation for lying

- `_classify_side_view()` - For left/right profile views

  - Calculates body angle from spine alignment
  - Uses shoulder-to-hip angle analysis
  - Considers ankle position relative to hips for sitting detection

- `_classify_diagonal_view()` - For diagonal camera angles

  - Combines frontal and side view analysis
  - Uses agreement between methods for confidence
  - Prioritizes lying detection for safety

- `_classify_general_view()` - Fallback method
  - Uses bounding box aspect ratios
  - Analyzes keypoint spread patterns
  - Provides robust classification when angle detection fails

#### **Improved Fall Detection**

- Enhanced `detect_fall()` method with multiple indicators:
  1. **Posture Transition**: Standing/sitting → lying
  2. **Rapid Vertical Movement**: Tracks downward movement via bounding box changes
  3. **Floor Proximity**: High floor proximity in lying position
  4. **Aspect Ratio Change**: Bounding box becoming wider (more horizontal)
- Requires at least 2 indicators for fall detection (reduces false positives)

### 2. Enhanced SimpleTracker Class (`simple_tracker.py`)

#### **Multi-Factor Tracking**

- Added enhanced tracking with 4 similarity factors:
  - **Position Similarity** (40%): Bounding box center distance
  - **Size Similarity** (20%): Bounding box area comparison
  - **Keypoint Similarity** (40%): Key body part position matching
  - **IoU Overlap** (backup): Traditional overlap measure

#### **Robust Keypoint Matching**

- `_calculate_keypoint_similarity()` focuses on stable keypoints:
  - Nose (head position)
  - Left/right shoulders
  - Left/right hips
- Calculates normalized distances between corresponding keypoints
- Handles partial keypoint visibility common in CCTV views

#### **Improved Track Management**

- Added disappeared frame counting for temporary occlusions
- Enhanced track cleanup for long-missing persons
- Better handling of detection-to-track assignments

## Technical Improvements

### **Multi-Angle Robustness**

1. **Bounding Box Analysis**: Uses width/height ratios for orientation detection
2. **Keypoint Visibility**: Adapts to partial visibility in different camera angles
3. **Relative Positioning**: Analyzes body part relationships regardless of viewpoint
4. **Movement Patterns**: Tracks movement trajectories for fall detection

### **CCTV-Specific Adaptations**

1. **Angle Tolerance**: Handles variations in camera mounting angles
2. **Occlusion Handling**: Robust to partial body visibility
3. **Distance Adaptation**: Normalizes measurements for different camera distances
4. **Lighting Compensation**: Relies on pose structure rather than appearance

## Configuration Parameters

### PoseAnalyzer Parameters

```python
lying_height_ratio_thresh = 0.4      # Height/width ratio for lying detection
bbox_aspect_ratio_lying = 2.0        # Width/height ratio threshold
vertical_distance_thresh = 0.3       # Normalized vertical distance threshold
keypoint_spread_thresh = 0.15        # Keypoint spread threshold
```

### SimpleTracker Parameters

```python
position_weight = 0.4                # Weight for position similarity
size_weight = 0.2                    # Weight for size similarity
keypoint_weight = 0.4               # Weight for keypoint similarity
max_position_distance = 100         # Max pixel distance for matching
max_size_ratio = 2.0                # Max size change ratio
max_disappeared = 10                # Max frames before track removal
```

## Benefits

1. **Improved Accuracy**: Better pose classification across different camera angles
2. **Reduced False Positives**: Enhanced fall detection with multiple indicators
3. **Robust Tracking**: Multi-factor similarity for better person tracking
4. **CCTV Optimization**: Designed specifically for fixed camera installations
5. **Angle Independence**: Works effectively regardless of camera mounting position

## Usage

The system automatically detects camera angles and applies appropriate algorithms. No manual configuration required for different camera positions. The enhanced logic handles:

- **Front-facing cameras**: In corridors, entrances
- **Back-facing cameras**: Behind patient areas
- **Side-view cameras**: Along room walls
- **Corner cameras**: Diagonal viewing angles
- **Ceiling cameras**: Top-down perspectives (partial support)

## Testing Recommendations

1. Test with video footage from different camera angles
2. Verify fall detection accuracy across viewing angles
3. Check person tracking consistency during movement
4. Validate pose classification in various lighting conditions
5. Monitor false positive/negative rates for each camera angle
