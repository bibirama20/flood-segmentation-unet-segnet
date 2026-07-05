import csv
import os

import gradio as gr
import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = 128
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "Image")
MASK_DIR = os.path.join(DATA_DIR, "Mask")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")

model_unet = tf.keras.models.load_model("unet_model.keras", compile=False)
model_segnet = tf.keras.models.load_model("segnet_model.keras", compile=False)


def load_image_to_mask_map():
    mapping = {}
    if not os.path.exists(METADATA_CSV):
        return mapping
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["Image"]] = row["Mask"]
    return mapping


IMAGE_TO_MASK = load_image_to_mask_map()


def overlay_mask(base_rgb, mask, color):
    color_mask = np.zeros_like(base_rgb)
    color_mask[:, :] = color
    alpha = 0.4
    mask3 = mask[:, :, None]
    return np.where(mask3 == 1, (base_rgb * (1 - alpha) + color_mask * alpha).astype(np.uint8), base_rgb)


def compute_metrics(y_true, y_pred_raw):
    y_true_f = y_true.flatten().astype(np.float64)
    y_pred_f = y_pred_raw.flatten().astype(np.float64)
    y_pred_round = (y_pred_f > 0.5).astype(np.float64)

    accuracy = float(np.mean(y_true_f == y_pred_round))

    dice = float((2 * np.sum(y_true_f * y_pred_f) + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1))

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = float((intersection + 1) / (union + 1))

    tp = np.sum(y_true_f * y_pred_round)
    fp = np.sum((1 - y_true_f) * y_pred_round)
    fn = np.sum(y_true_f * (1 - y_pred_round))
    precision = float(tp / (tp + fp + 1e-7))
    recall = float(tp / (tp + fn + 1e-7))
    f1 = float(2 * precision * recall / (precision + recall + 1e-7))

    return accuracy, dice, iou, precision, recall, f1


def read_and_binarize_mask(mask_path):
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
    return (gt / 255.0 > 0.5).astype(np.float32)


def load_dataset_ground_truth(filename):
    mask_filename = IMAGE_TO_MASK.get(filename)
    if mask_filename is None:
        return None
    mask_path = os.path.join(MASK_DIR, mask_filename)
    if not os.path.exists(mask_path):
        return None
    return read_and_binarize_mask(mask_path)


def segment(image_path, mask_path):
    zero_metrics = (0, 0, 0, 0, 0, 0)
    if image_path is None:
        return (None, None, None, None, "") + zero_metrics + zero_metrics

    image_bgr = cv2.imread(image_path)
    resized_bgr = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    display_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    model_input = np.expand_dims(resized_bgr / 255.0, axis=0)

    pred_unet = model_unet.predict(model_input, verbose=0)[0]
    pred_segnet = model_segnet.predict(model_input, verbose=0)[0]

    mask_unet = (pred_unet > 0.5).astype(np.uint8).squeeze()
    mask_segnet = (pred_segnet > 0.5).astype(np.uint8).squeeze()

    mask_unet_img = (mask_unet * 255).astype(np.uint8)
    mask_segnet_img = (mask_segnet * 255).astype(np.uint8)

    overlay_unet = overlay_mask(display_rgb, mask_unet, [255, 0, 0])   # merah
    combined = overlay_unet.copy()
    combined[mask_segnet == 1] = [0, 0, 255]  # biru = SegNet

    if mask_path is not None:
        ground_truth = read_and_binarize_mask(mask_path)
        source = "mask yang Anda upload"
    else:
        ground_truth = load_dataset_ground_truth(os.path.basename(image_path))
        source = "dataset"

    if ground_truth is None:
        metrics_unet = zero_metrics
        metrics_segnet = zero_metrics
        status = (
            "⚠️ Tidak ada ground truth untuk citra ini, sehingga metrik di bawah ditampilkan "
            "sebagai 0. Upload mask anotasi manual asli (BUKAN mask hasil prediksi U-Net/SegNet "
            "di atas) lewat kolom **\"Upload Mask Ground Truth\"**, lalu jalankan ulang untuk "
            "melihat nilai metriknya."
        )
    else:
        metrics_unet = compute_metrics(ground_truth, pred_unet.squeeze())
        metrics_segnet = compute_metrics(ground_truth, pred_segnet.squeeze())
        status = f"✅ Ground truth ditemukan dari {source} — metrik dihitung dari perbandingan asli."

    return (mask_unet_img, mask_segnet_img, combined, display_rgb, status) + metrics_unet + metrics_segnet


with gr.Blocks(title="Perbandingan Segmentasi U-Net vs SegNet") as demo:
    gr.Markdown("# Perbandingan Segmentasi Citra Banjir: U-Net vs SegNet")
    gr.Markdown("Upload citra untuk melihat dan membandingkan hasil segmentasi kedua model.")

    with gr.Tab("Segmentasi"):
        with gr.Row():
            input_image = gr.Image(type="filepath", label="Upload Citra")
            input_mask = gr.Image(
                type="filepath",
                label="Upload Mask Ground Truth (opsional, untuk citra di luar dataset)",
            )
        gr.Markdown(
            "Upload mask **yang benar** (gambar/anotasi manual area banjir asli), "
            "bukan hasil prediksi model."
        )
        run_button = gr.Button("Jalankan Segmentasi", variant="primary")

        with gr.Row():
            output_original = gr.Image(label="Citra (128x128)")
            output_unet = gr.Image(label="Mask U-Net")
            output_segnet = gr.Image(label="Mask SegNet")
            output_overlay = gr.Image(label="Overlay (Merah=U-Net, Biru=SegNet)")
        gr.Markdown(
            "⚠️ Mask U-Net dan Mask SegNet di atas cuma **tebakan model**, jangan diupload "
            "sebagai ground truth."
        )

    with gr.Tab("Metrik Model"):
        gr.Markdown("### Metrik Evaluasi (dihitung otomatis dari ground truth dataset)")
        status_message = gr.Markdown("")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**U-Net**")
                metric_accuracy_unet = gr.Number(label="Accuracy", value=0)
                metric_dice_unet = gr.Number(label="Dice", value=0)
                metric_iou_unet = gr.Number(label="IoU", value=0)
                metric_precision_unet = gr.Number(label="Precision", value=0)
                metric_recall_unet = gr.Number(label="Recall", value=0)
                metric_f1_unet = gr.Number(label="F1", value=0)
            with gr.Column():
                gr.Markdown("**SegNet**")
                metric_accuracy_segnet = gr.Number(label="Accuracy", value=0)
                metric_dice_segnet = gr.Number(label="Dice", value=0)
                metric_iou_segnet = gr.Number(label="IoU", value=0)
                metric_precision_segnet = gr.Number(label="Precision", value=0)
                metric_recall_segnet = gr.Number(label="Recall", value=0)
                metric_f1_segnet = gr.Number(label="F1", value=0)
        gr.Markdown(
            "_Ground truth diambil dari mask yang diupload manual di tab Segmentasi (jika ada), atau "
            "dicocokkan otomatis dari dataset lewat `data/metadata.csv`. Jika tidak ada ground truth "
            "sama sekali, metrik akan tetap 0._"
        )

    run_button.click(
        fn=segment,
        inputs=[input_image, input_mask],
        outputs=[
            output_unet, output_segnet, output_overlay, output_original, status_message,
            metric_accuracy_unet, metric_dice_unet, metric_iou_unet,
            metric_precision_unet, metric_recall_unet, metric_f1_unet,
            metric_accuracy_segnet, metric_dice_segnet, metric_iou_segnet,
            metric_precision_segnet, metric_recall_segnet, metric_f1_segnet,
        ],
    )

if __name__ == "__main__":
    demo.launch(show_api=False)
