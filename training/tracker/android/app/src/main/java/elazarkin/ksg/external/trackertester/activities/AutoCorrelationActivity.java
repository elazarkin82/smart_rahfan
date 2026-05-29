package elazarkin.ksg.external.trackertester.activities;

import android.Manifest;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;

import elazarkin.ksg.external.trackertester.MainActivity;
import elazarkin.ksg.external.trackertester.R;
import elazarkin.ksg.external.trackertester.base.camera.CameraHelper;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class AutoCorrelationActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1001;

    // UI Elements
    private PreviewView viewFinder;
    private ImageView capturedImageView;
    private TextView tutorialHud;
    private TextView lblStatus;
    private Button btnReset;
    private Button btnBack;
    
    // Results & Telemetry Elements
    private LinearLayout resultsPanel;
    private ImageView heatmapImageView;
    private TextView txtLatency;
    private TextView txtSelectedCoords;
    private TextView txtPredictedCoords;
    private TextView txtOffsetError;
    private ImageView cropHistView;
    private ImageView cropPrevView;
    private ImageView cropCurrView;

    // Camera State
    private CameraHelper cameraHelper;
    private Bitmap capturedBitmap = null;
    private boolean isCaptured = false;

    // TFLite State
    private Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_auto_correlation);

        // 1. Initialize Views
        viewFinder = findViewById(R.id.viewFinder);
        capturedImageView = findViewById(R.id.capturedImageView);
        tutorialHud = findViewById(R.id.tutorial_hud);
        lblStatus = findViewById(R.id.lbl_status);
        btnReset = findViewById(R.id.btn_reset);
        btnBack = findViewById(R.id.btn_back);
        
        resultsPanel = findViewById(R.id.results_panel);
        heatmapImageView = findViewById(R.id.heatmapImageView);
        txtLatency = findViewById(R.id.txt_latency);
        txtSelectedCoords = findViewById(R.id.txt_selected_coords);
        txtPredictedCoords = findViewById(R.id.txt_predicted_coords);
        txtOffsetError = findViewById(R.id.txt_offset_error);
        cropHistView = findViewById(R.id.cropHistView);
        cropPrevView = findViewById(R.id.cropPrevView);
        cropCurrView = findViewById(R.id.cropCurrView);

        // 2. Load TFLite Model
        try {
            tflite = new Interpreter(loadModelFile());
            lblStatus.setText("Status: Model loaded successfully");
        } catch (Exception e) {
            e.printStackTrace();
            lblStatus.setText("Status: Failed to load TFLite model!");
            Toast.makeText(this, "TFLite Model Load Failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }

        // 3. Set Up Button Click Listeners
        btnBack.setOnClickListener(v -> finish());

        btnReset.setOnClickListener(v -> resetTracker());

        // 4. Set Up Touch Listeners for Snapshot and Selection
        setupTouchInteractions();

        // 5. Initialize Camera Helper
        cameraHelper = new CameraHelper(this, viewFinder);
        if (cameraHelper.hasCameraPermission()) {
            cameraHelper.startCamera(new CameraHelper.OnCameraReadyCallback() {
                @Override
                public void onCameraReady() {
                    lblStatus.setText("Status: Camera active. Point and capture!");
                }

                @Override
                public void onCameraError(Exception e) {
                    lblStatus.setText("Status: Failed to start camera preview.");
                }
            });
        } else {
            cameraHelper.requestCameraPermission(this, CAMERA_PERMISSION_REQUEST_CODE);
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        // 1. Try direct memory-mapping from assets first
        try {
            AssetFileDescriptor fileDescriptor = this.getAssets().openFd("tracker_model.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            // 2. Fallback: Copy compressed model from assets to local cache dir, then load it from there
            java.io.File tempFile = new java.io.File(getCacheDir(), "temp_tracker_model.tflite");
            if (!tempFile.exists()) {
                try (java.io.InputStream in = getAssets().open("tracker_model.tflite");
                     java.io.OutputStream out = new java.io.FileOutputStream(tempFile)) {
                    byte[] buffer = new byte[4096];
                    int read;
                    while ((read = in.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                }
            }
            FileInputStream inputStream = new FileInputStream(tempFile);
            FileChannel fileChannel = inputStream.getChannel();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, tempFile.length());
        }
    }

    private void setupTouchInteractions() {
        // Screen Touch to Trigger Snapshot on viewFinder
        viewFinder.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                if (!isCaptured) {
                    takeSnapshot();
                    return true;
                }
            }
            return false;
        });

        // Screen Touch to Select Target Coordinate on capturedImageView
        capturedImageView.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                if (isCaptured && capturedBitmap != null) {
                    // Extract normalized touch coordinates [0, 1] relative to the captured Bitmap
                    float viewX = event.getX();
                    float viewY = event.getY();
                    
                    // Account for FIT_CENTER scaling inside ImageView
                    float[] imageCoords = getNormalizedImageCoords(viewX, viewY);
                    if (imageCoords != null) {
                        float tx = imageCoords[0];
                        float ty = imageCoords[1];
                        
                        runTrackingInference(tx, ty);
                        return true;
                    }
                }
            }
            return false;
        });
    }

    private void takeSnapshot() {
        lblStatus.setText("Status: Grabbing frame...");
        Bitmap bitmap = cameraHelper.captureFrame();
        if (bitmap != null) {
            capturedBitmap = bitmap;
            isCaptured = true;
            
            // Toggle visual views
            viewFinder.setVisibility(View.GONE);
            capturedImageView.setVisibility(View.VISIBLE);
            capturedImageView.setImageBitmap(capturedBitmap);
            
            btnReset.setVisibility(View.VISIBLE);
            tutorialHud.setText("Tap on the object you want to track");
            lblStatus.setText("Status: Frame captured. Select target!");
        } else {
            lblStatus.setText("Status: Failed to grab frame!");
            Toast.makeText(this, "Failed to capture preview frame", Toast.LENGTH_SHORT).show();
        }
    }

    private void runTrackingInference(float tx, float ty) {
        if (tflite == null) {
            Toast.makeText(this, "Model not initialized", Toast.LENGTH_SHORT).show();
            return;
        }

        lblStatus.setText("Status: Running FCN inference...");
        
        long startTime = SystemClock.elapsedRealtime();

        // 1. Allocate input arrays for TargetTracker3 (flat 256*256*2 floats for 2 channels)
        float[] histBuffer = new float[256 * 256 * 2];
        float[] prevBuffer = new float[256 * 256 * 2];
        float[] currBuffer = new float[256 * 256 * 2];

        long preStart = SystemClock.elapsedRealtime();
        // 2. Invoke JNI Preprocessing: Downsample full image (no crop!)
        // - hist_frame: Gaussian soft mask (sigma = 30.0f) centered at tx, ty
        MainActivity.downsampleAndMaskFrameV3(capturedBitmap, tx, ty, 128.0f, true, 30.0f, false, histBuffer);
        // - prev_frame: Gaussian soft mask (sigma = 30.0f) centered at tx, ty
        MainActivity.downsampleAndMaskFrameV3(capturedBitmap, tx, ty, 50.0f, true, 30.0f, false, prevBuffer);
        // - curr_frame: search frame (no crop, zeros attention mask)
        MainActivity.downsampleAndMaskFrameV3(capturedBitmap, tx, ty, 0.0f, false, 0.0f, true, currBuffer);
        long preDuration = SystemClock.elapsedRealtime() - preStart;

        // 3. Prepare inputs/outputs for TFLite multiple inputs execution
        // Reshape flat float arrays to [1][256][256][2] expected by model
        float[][][][] histInput = new float[1][256][256][2];
        float[][][][] prevInput = new float[1][256][256][2];
        float[][][][] currInput = new float[1][256][256][2];

        for (int i = 0; i < 256 * 256; i++) {
            int y = i / 256;
            int x = i % 256;
            // Channel 0: Downsampled unmasked grayscale
            histInput[0][y][x][0] = histBuffer[2 * i];
            prevInput[0][y][x][0] = prevBuffer[2 * i];
            currInput[0][y][x][0] = currBuffer[2 * i];
            
            // Channel 1: Attention mask
            histInput[0][y][x][1] = histBuffer[2 * i + 1];
            prevInput[0][y][x][1] = prevBuffer[2 * i + 1];
            currInput[0][y][x][1] = currBuffer[2 * i + 1];
        }

        Object[] inputs = new Object[]{ histInput, prevInput, currInput };
        float[][][][] outputHeatmap = new float[1][64][64][1];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputHeatmap);

        // 4. Execute inference
        long infStart = SystemClock.elapsedRealtime();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long infDuration = SystemClock.elapsedRealtime() - infStart;

        // 5. Flatten the 64x64 output heatmap for JNI postprocessing
        float[] flatHeatmap = new float[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                flatHeatmap[y * 64 + x] = outputHeatmap[0][y][x][0];
            }
        }

        // 6. Invoke JNI Center of Mass (threshold = 0.1f)
        long postStart = SystemClock.elapsedRealtime();
        float[] predCoords = MainActivity.calculateCenterOfMass(flatHeatmap, 0.1f);
        long postDuration = SystemClock.elapsedRealtime() - postStart;

        long totalDuration = SystemClock.elapsedRealtime() - startTime;

        // Direct Coordinate Mapping: Since we downsample the FULL frame (no crop),
        // the Center of Mass normalized position is ALREADY the absolute normalized target position!
        float px = predCoords[0];
        float py = predCoords[1];

        // 8. Render Results on Screen
        renderOutputs(tx, ty, px, py, flatHeatmap, histBuffer, prevBuffer, currBuffer);

        // 9. Update Telemetry text
        txtLatency.setText(String.format("Latency: JNI Pre:%dms | TFLite:%dms | JNI Post:%dms (Total:%dms)", 
                preDuration, infDuration, postDuration, totalDuration));
        txtSelectedCoords.setText(String.format("Selected Coordinate: (%.3f, %.3f)", tx, ty));
        txtPredictedCoords.setText(String.format("Predicted CoM: (%.3f, %.3f)", px, py));
        
        // Calculate Euclidean offset error in pixels on the source image
        float errX = (px - tx) * capturedBitmap.getWidth();
        float errY = (py - ty) * capturedBitmap.getHeight();
        double errorPx = Math.sqrt(errX * errX + errY * errY);
        txtOffsetError.setText(String.format("Auto-Correlation Error: %.2f pixels", errorPx));

        lblStatus.setText("Status: Inference completed successfully!");
        tutorialHud.setText("Done! Select another point or press Reset");
    }

    private void renderOutputs(float tx, float ty, float px, float py, 
                               float[] heatmap, float[] hist, float[] prev, float[] curr) {
        
        // A. Draw circles on a mutable copy of the captured Bitmap
        Bitmap mutableBitmap = capturedBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        
        int w = mutableBitmap.getWidth();
        int h = mutableBitmap.getHeight();
        
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5.0f);
        paint.setAntiAlias(true);
        
        // Draw selected target (Yellow circle with inner dot)
        paint.setColor(Color.YELLOW);
        canvas.drawCircle(tx * w, ty * h, 18.0f, paint);
        paint.setStyle(Paint.Style.FILL);
        canvas.drawCircle(tx * w, ty * h, 5.0f, paint);
        
        // Draw predicted coordinate (Neon Green circle with inner dot)
        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.GREEN);
        canvas.drawCircle(px * w, py * h, 18.0f, paint);
        paint.setStyle(Paint.Style.FILL);
        canvas.drawCircle(px * w, py * h, 5.0f, paint);
        
        capturedImageView.setImageBitmap(mutableBitmap);

        // B. Render Heatmap (Colorized Jet: Red for hot, Blue for cold)
        Bitmap hmBitmap = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
        int[] hmColors = new int[64 * 64];
        for (int i = 0; i < 64 * 64; i++) {
            float val = heatmap[i];
            val = Math.max(0.0f, Math.min(val, 1.0f));
            
            // Basic Heat colormap mapping (R: red, G: green, B: blue)
            int r = (int)(val * 255.0f);
            int b = (int)((1.0f - val) * 255.0f);
            int g = (int)(val * 100.0f); // Add a touch of green for smoother peak transition
            
            hmColors[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        hmBitmap.setPixels(hmColors, 0, 64, 0, 0, 64, 64);
        heatmapImageView.setImageBitmap(hmBitmap);

        // C. Render Grayscale Visual Debug Crops (256x256)
        cropHistView.setImageBitmap(renderGrayscaleCrop(hist));
        cropPrevView.setImageBitmap(renderGrayscaleCrop(prev));
        cropCurrView.setImageBitmap(renderGrayscaleCrop(curr));

        // Make the results panel visible
        resultsPanel.setVisibility(View.VISIBLE);
    }

    private Bitmap renderGrayscaleCrop(float[] floatBuffer) {
        Bitmap cropBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
        int[] colors = new int[256 * 256];
        for (int i = 0; i < 256 * 256; i++) {
            // Channel 0 is the unmasked grayscale image
            int grayVal = (int)(floatBuffer[2 * i] * 255.0f);
            grayVal = Math.max(0, Math.min(grayVal, 255));
            
            // Channel 1 is the attention mask
            float maskVal = floatBuffer[2 * i + 1];
            
            // Premium feature: blend the unmasked image with a glowing red overlay for the attention mask
            int r = grayVal;
            int g = grayVal;
            int b = grayVal;
            
            if (maskVal > 0.01f) {
                float alpha = 0.35f * maskVal; // max 35% opacity
                r = (int) (grayVal * (1.0f - alpha) + 255.0f * alpha);
                g = (int) (grayVal * (1.0f - alpha));
                b = (int) (grayVal * (1.0f - alpha));
            }
            
            colors[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        cropBitmap.setPixels(colors, 0, 256, 0, 0, 256, 256);
        return cropBitmap;
    }

    private float[] getNormalizedImageCoords(float viewX, float viewY) {
        // Calculate image aspect ratio scaling inside FIT_CENTER ImageView
        int viewWidth = capturedImageView.getWidth();
        int viewHeight = capturedImageView.getHeight();
        int imgWidth = capturedBitmap.getWidth();
        int imgHeight = capturedBitmap.getHeight();
        
        float viewRatio = (float) viewWidth / viewHeight;
        float imgRatio = (float) imgWidth / imgHeight;
        
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        
        if (imgRatio > viewRatio) { // Fit Width, height is letterboxed
            float actualHeight = viewWidth / imgRatio;
            offsetY = (viewHeight - actualHeight) / 2.0f;
            scaleX = (float) imgWidth / viewWidth;
            scaleY = (float) imgHeight / actualHeight;
        } else { // Fit Height, width is letterboxed
            float actualWidth = viewHeight * imgRatio;
            offsetX = (viewWidth - actualWidth) / 2.0f;
            scaleX = (float) imgWidth / actualWidth;
            scaleY = (float) imgHeight / viewHeight;
        }
        
        float bmpX = (viewX - offsetX) * scaleX;
        float bmpY = (viewY - offsetY) * scaleY;
        
        if (bmpX >= 0 && bmpX < imgWidth && bmpY >= 0 && bmpY < imgHeight) {
            return new float[]{ bmpX / imgWidth, bmpY / imgHeight };
        }
        return null; // Click fell outside bitmap bounds (on letterbox area)
    }

    private void resetTracker() {
        isCaptured = false;
        capturedBitmap = null;
        
        viewFinder.setVisibility(View.VISIBLE);
        capturedImageView.setVisibility(View.GONE);
        resultsPanel.setVisibility(View.GONE);
        btnReset.setVisibility(View.GONE);
        
        tutorialHud.setText("Tap screen to take a snapshot");
        lblStatus.setText("Status: Camera active. Point and capture!");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (cameraHelper.hasCameraPermission()) {
                cameraHelper.startCamera(new CameraHelper.OnCameraReadyCallback() {
                    @Override
                    public void onCameraReady() {
                        lblStatus.setText("Status: Camera active. Point and capture!");
                    }

                    @Override
                    public void onCameraError(Exception e) {
                        lblStatus.setText("Status: Failed to start camera preview.");
                    }
                });
            } else {
                Toast.makeText(this, "Camera permission is required to capture images for tests.", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onDestroy() {
        if (tflite != null) {
            tflite.close();
        }
        if (cameraHelper != null) {
            cameraHelper.shutdown();
        }
        super.onDestroy();
    }
}
