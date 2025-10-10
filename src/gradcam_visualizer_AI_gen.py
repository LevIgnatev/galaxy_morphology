def generate_gradcam_overlay_from_array(model_path: str, image_rgb_224_np):
    """
    Wrapper that uses the same Grad-CAM pipeline defined above,
    but takes a (224,224,3) RGB float/uint8 array and returns (overlay_uint8, softmax_probs).
    """
    import numpy as _np, cv2 as _cv2, tensorflow as _tf

    num_classes = _tf.keras.models.load_model(model_path, compile=False).output_shape[-1]

    inp = _tf.keras.Input(shape=(224, 224, 3), name="input")
    x = _tf.keras.applications.resnet50.preprocess_input(inp)
    resnet_no_pool = _tf.keras.applications.ResNet50(include_top=False, weights="imagenet", pooling=None)
    resnet_no_pool.trainable = False
    conv_feat = resnet_no_pool(x)
    gap = _tf.keras.layers.GlobalAveragePooling2D(name="gap")(conv_feat)
    dense1 = _tf.keras.layers.Dense(128, activation="relu", name="dense_clone")
    drop1  = _tf.keras.layers.Dropout(0.2, name="dropout_clone")
    head   = _tf.keras.layers.Dense(num_classes, activation="softmax", name="head_clone")(drop1(dense1(gap)))
    cam_model = _tf.keras.Model(inp, [conv_feat, head], name="gradcam_ready")

    orig = _tf.keras.models.load_model(model_path, compile=False)
    orig_denses = [l for l in orig.layers if isinstance(l, _tf.keras.layers.Dense)]
    dense1.set_weights(orig_denses[0].get_weights())
    cam_model.get_layer("head_clone").set_weights(orig_denses[1].get_weights())

    x_in = image_rgb_224_np.astype("float32")[None, ...]

    with _tf.GradientTape() as tape:
        conv_out, preds = cam_model(x_in, training=False)
        class_idx = _tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads   = tape.gradient(loss, conv_out)
    weights = _tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam     = _tf.nn.relu(_tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    cam = _cv2.resize(cam, (224, 224), interpolation=_cv2.INTER_LINEAR)

    heat = _cv2.applyColorMap((cam * 255).astype(_np.uint8), _cv2.COLORMAP_JET)
    heat = _cv2.cvtColor(heat, _cv2.COLOR_BGR2RGB)
    overlay = _np.clip(0.5 * image_rgb_224_np + 0.5 * heat, 0, 255).astype(_np.uint8)
    return overlay, preds.numpy()[0]
