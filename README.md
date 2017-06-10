# markerless-augmentation

1. Camera motion tracking pipeline is detailed in: `camera_pose_from_images.py`
    * `compute_correspondences.py`: has utility functions for computing features, descriptors, matches and refined matches (correspondence) between corresponding views
    * `compute_fundamental_matrix.py`: has functions for computing the fundamental matrix from normalized eight-point algorithm
    * `compute_camera_pose.py`: has functions for computing camera pose between corresponding views by estimating the pose, then computing 3D points from the correspondences using non-linear triangulation, then computing essential matrix from camera intrinsics and fundamental matrix, and finally computing the camera pose from the SVD of Essential matrix and resolving the ambiguity.

2. Virtual Object Placement is attempted in: `read_video_select_points.py`
    * This script loads a video frame, opens up a window context, draw the frame as a texture on a quad and listens for user's clicks on the frame to designate world coordinate system.
    * We then use the functions detailed in camera motion tracking pipeline to get the correspondence of these points, compute 3D points using triangulation and to compute object pose in camera frame.

3. Marker tracking and pose estimation is implemented in: `read_marker.py`
    * Here, ArUco markers are detected in the scene, their corners extracted and camera pose determined.
