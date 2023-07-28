while f == 1:
    #print('{:.1f}: {}'.format(number_of_samples*sample_interval, "motion detected" if m.motion_detected() else "-"))
    if m.motion_detected:
        dave_detected = True
    sleep(sample_interval)
    number_of_samples = number_of_samples + 1
    level = 0
    brightness = 3
    blue_to_red = 1
    # set the level on the led bar
    
    ### code detects a pose where the hands and wrists are near the waste###




    # Preprocessing function
    def preprocess_image(img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height

        # Padding
        xpad = 0
        ypad = 20
        canvas = 255 * np.ones((height + 2 * ypad, width + 2 * xpad, 3), dtype=img.dtype)
        canvas[ypad:height + ypad, xpad:width + xpad, :] = img
        img = canvas

        # Crop
        new_height = shapeIn[1]
        img_resized = cv2.resize(img, (int(aspect_ratio * new_height), new_height))
        img_cropped = img_resized[0:shapeIn[1], 0:shapeIn[2], :].reshape((shapeIn[1], shapeIn[2], 3))
        img_cropped = img_cropped.astype(np.float32) / 255.0  # Make it floats

        return img_cropped

    # Run Model function
    def run_model(img_cropped):
        processed_image = img_cropped
        processed_image = processed_image.reshape(shapeIn).astype(np.float32, order="C")
        output_data = [np.zeros(shapeOut, dtype=np.float32, order="C")]
        input_data = [np.zeros(shapeIn, dtype=np.float32, order="C")]
        input_data[0] = processed_image
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)
        return output_data[0]

    # Updated find_keypoints function
    def find_keypoints(output_data, threshold=0.1):
        keypoints = []
        for i in range(18):  # There are 18 keypoints in OpenPose
            keypoints_map = output_data[0][:, :, i]
            max_value = np.max(keypoints_map)
            if max_value > threshold:
                location = np.where(keypoints_map == max_value)
                keypoints.append((location[1][0], location[0][0]))
            else:
                keypoints.append(None)
        return keypoints

    # Function to check for potential gun-holding pose
    # Updated detect_gun_pose function with increased threshold
    def detect_gun_pointing_pose(keypoints, threshold=50):
        left_hand = keypoints[7] if keypoints[7] is not None else None  # Left Wrist
        right_hand = keypoints[4] if keypoints[4] is not None else None  # Right Wrist
        left_shoulder = keypoints[5] if keypoints[5] is not None else None  # Left Shoulder
        right_shoulder = keypoints[2] if keypoints[2] is not None else None  # Right Shoulder

        if left_hand and left_shoulder:
            left_arm_length = np.linalg.norm(np.array(left_hand) - np.array(left_shoulder))
            left_wrist_height = keypoints[7][1] if keypoints[7] is not None else 0
            left_shoulder_height = keypoints[5][1] if keypoints[5] is not None else 0

            if left_arm_length < threshold and left_wrist_height > left_shoulder_height:
                print("Warning: Gun pointing pose detected (Left Hand)")
                

        if right_hand and right_shoulder:
            right_arm_length = np.linalg.norm(np.array(right_hand) - np.array(right_shoulder))
            right_wrist_height = keypoints[4][1] if keypoints[4] is not None else 0
            right_shoulder_height = keypoints[2][1] if keypoints[2] is not None else 0

            if right_arm_length < threshold and right_wrist_height > right_shoulder_height:
                print("Warning: Gun pointing pose detected (Right Hand)")




    overlay = DpuOverlay("dpu.bit")
    archive_filename = "openpose_pruned_0_3.tar.gz"
    extracted_model_path = "openpose_pruned_0_3.xmodel"
    # The following will download the model if it is not already present in the folder:
    # model_download_url = "https://www.xilinx.com/bin/public/openDownload?filename=openpose_pruned_0_3-zcu102_zcu104_kv260-r2.5.0.tar.gz"
    # os.system("wget -nv -O \"{}\" \"{}\"".format(archive_filename, model_download_url))
    # os.system("tar -xvf \"{}\"".format(archive_filename))
    overlay.load_model(extracted_model_path)

    # Get the input shape of the model
    dpu = overlay.runner
    inputTensors = dpu.get_input_tensors()
    shapeIn = tuple(inputTensors[0].dims)
    outputTensors = dpu.get_output_tensors()
    shapeOut = tuple(outputTensors[0].dims)

    # Real-time Webcam Pose Estimation with Gun Detection (Updated)
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(3, shapeIn[1])
    capture.set(4, shapeIn[2])

    try:
        while True:
            success, newimg = capture.read()
            if not success:
                print("Error reading webcam image.")
                break

            img_cropped = preprocess_image(newimg)
            output = run_model(img_cropped)

            # Extract keypoints and their corresponding body part names from the output
            keypoints = find_keypoints(output)

            # Clear the previous plot
            plt.clf()

            # Display the keypoints along with the heatmap
            plt.imshow(img_cropped)
            heatmap = np.squeeze(np.max(output[0][:][:][:], axis=2))
            heatmap = cv2.resize(heatmap, (shapeIn[2], shapeIn[1]))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest', alpha=0.4)

            # Check for potential gun-holding pose
            detect_gun_pointing_pose(keypoints)

            # Display the updated frame
            IPython.display.clear_output(wait=True)
            IPython.display.display(plt.gcf())

    except KeyboardInterrupt:
        print("Stopped by user.")

    # Release the webcam
    capture.release()
    cv2.destroyAllWindows()

    
    while motionDave or poseDave == True:
        print(2)
        for i in range(1,11):
            level = i
            ledbar.set_level(int(level), brightness, blue_to_red)
            sleep(0.1)
        if level  != 0:
            level = 0
        sleep(0.1)
        while poseDave or objectDave  == True:
            for x in range(1,11):
                level = x
                ledbar.set_level(int(level), brightness, blue_to_red)
                sleep(0.01)
            if level !=0:
                level = 0
            sleep(0.01)
    level = 0
    ledbar.set_level(int(level), brightness, blue_to_red)
