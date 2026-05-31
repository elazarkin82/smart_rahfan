package elazarkin.ksg.tracker3_lite.activities;

import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
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
import android.content.res.AssetFileDescriptor;

import elazarkin.ksg.tracker3_lite.MainActivity;
import elazarkin.ksg.tracker3_lite.R;
import elazarkin.ksg.tracker3_lite.base.camera.CameraHelper;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class FrameStreamActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1002;
    private static final float CONFIDENCE_THRESHOLD = 0.20f;

    // UI States
    private static final int STATE_IDLE = 0;
    private static final int STATE_TRACKING = 1;
    private static final int STATE_LOST = 2;

    private int currentUiState = STATE_IDLE;

    // UI Elements
    private PreviewView viewFinder;
    private ImageView capturedImageView;
    private TextView tutorialHud;
    private TextView lblStatus;
    private Button btnReset;
    private Button btnBack;

    // Results & Telemetry
    private LinearLayout resultsPanel;
    private ImageView heatmapImageView;
    private TextView txtLatency;
    private TextView txtPredictedCoords;
    private TextView txtBufferStatus;
    private ImageView cropHistView;
    private ImageView cropCurrView;

    // Camera and Threading State
    private CameraHelper cameraHelper;
    private boolean isLoopActive = false;
    private final Handler trackingHandler = new Handler(Looper.getMainLooper());
    
    // Template Anchor State
    private Bitmap histBitmap = null;
    private float histX = 0.5f;
    private float histY = 0.5f;
    private float[] histBuffer = new float[256 * 256 * 2]; // Grayscale + Mask

    // TFLite State
    private Interpreter tflite;

    // Repeating loop to capture and process frames at 30 FPS
    private final Runnable trackingRunnable = new Runnable() {
        @Override
        public void run() {
            if (isLoopActive && currentUiState == STATE_TRACKING) {
                processNextFrame();
            }
            if (isLoopActive) {
                trackingHandler.postDelayed(this, 33); // ~30 FPS
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_frame_stream);

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
        txtPredictedCoords = findViewById(R.id.txt_predicted_coords);
        txtBufferStatus = findViewById(R.id.txt_buffer_status);
        cropHistView = findViewById(R.id.cropHistView);
        cropCurrView = findViewById(R.id.cropCurrView);

        // 2. Load TFLite Model
        try {
            tflite = new Interpreter(loadModelFile());
            lblStatus.setText("Status: Engine loaded successfully");
        } catch (Exception e) {
            e.printStackTrace();
            lblStatus.setText("Status: Failed to load TFLite model!");
            Toast.makeText(this, "TFLite Model Load Failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }

        // 3. Buttons
        btnBack.setOnClickListener(v -> finish());
        btnReset.setOnClickListener(v -> resetTrackerToIdle());

        // 4. Screen Touch for target selection
        setupTouchInteractions();

        // 5. Initialize Camera Helper
        cameraHelper = new CameraHelper(this, viewFinder);
        if (cameraHelper.hasCameraPermission()) {
            startCameraStream();
        } else {
            cameraHelper.requestCameraPermission(this, CAMERA_PERMISSION_REQUEST_CODE);
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        try {
            AssetFileDescriptor fileDescriptor = this.getAssets().openFd("tracker_model.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
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
        capturedImageView.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                if (currentUiState == STATE_IDLE) {
                    float viewX = event.getX();
                    float viewY = event.getY();
                    
                    Bitmap currentFrame = cameraHelper.captureFrame();
                    if (currentFrame != null) {
                        float[] imageCoords = getNormalizedCoords(viewX, viewY);
                        if (imageCoords != null) {
                            initializeTrackingAt(imageCoords[0], imageCoords[1], currentFrame);
                            return true;
                        }
                    }
                }
            }
            return false;
        });
    }

    private void startCameraStream() {
        lblStatus.setText("Status: Connecting camera...");
        cameraHelper.startCamera(new CameraHelper.OnCameraReadyCallback() {
            @Override
            public void onCameraReady() {
                lblStatus.setText("Status: Active. Select target to track!");
                tutorialHud.setText("Tap screen to lock onto an object");
                
                isLoopActive = true;
                trackingHandler.post(trackingRunnable);
            }

            @Override
            public void onCameraError(Exception e) {
                lblStatus.setText("Status: Camera bind failed!");
            }
        });
    }

    private void initializeTrackingAt(float tx, float ty, Bitmap currentFrame) {
        histBitmap = currentFrame;
        histX = tx;
        histY = ty;

        // 1. Preprocess anchored frame once: Grayscale + Exponential Cone Mask (2 channels)
        MainActivity.downsampleAndMaskFrameV3(histBitmap, histX, histY, 128.0f, true, 30.0f, false, 2, histBuffer);

        // 2. Set UI State
        currentUiState = STATE_TRACKING;
        btnReset.setVisibility(View.VISIBLE);
        resultsPanel.setVisibility(View.VISIBLE);
        tutorialHud.setText("Active tracking loop running at 30 FPS");
        tutorialHud.setTextColor(Color.parseColor("#00e6ff"));
        lblStatus.setText("Status: Active tracking");

        drawTrackingIndicator(tx, ty, currentFrame);
    }

    private void processNextFrame() {
        if (tflite == null || currentUiState != STATE_TRACKING || histBitmap == null) return;

        Bitmap currentFrame = cameraHelper.captureFrame();
        if (currentFrame == null) return;

        long startTime = SystemClock.elapsedRealtime();

        // 1. Preprocess search frame: Grayscale only (1 channel, no mask)
        float[] currBuffer = new float[256 * 256 * 1];
        long preStart = SystemClock.elapsedRealtime();
        MainActivity.downsampleAndMaskFrameV3(currentFrame, 0.0f, 0.0f, 0.0f, false, 0.0f, true, 1, currBuffer);
        long preDuration = SystemClock.elapsedRealtime() - preStart;

        // 2. Assemble 4D input tensors
        float[][][][] histInput = new float[1][256][256][2];
        float[][][][] currInput = new float[1][256][256][1];

        for (int i = 0; i < 256 * 256; i++) {
            int y = i / 256;
            int x = i % 256;
            histInput[0][y][x][0] = histBuffer[2 * i];
            histInput[0][y][x][1] = histBuffer[2 * i + 1];
        }

        for (int i = 0; i < 256 * 256; i++) {
            int y = i / 256;
            int x = i % 256;
            currInput[0][y][x][0] = currBuffer[i];
        }

        Object[] inputs = new Object[]{ histInput, currInput };
        float[][][][] outputHeatmap = new float[1][64][64][1];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputHeatmap);

        // 3. Execute TFLite Inference
        long infStart = SystemClock.elapsedRealtime();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long infDuration = SystemClock.elapsedRealtime() - infStart;

        // 4. Inspect predicted heatmap activation peak (Confidence Check)
        float maxConfidence = 0.0f;
        float[] flatHeatmap = new float[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                float val = outputHeatmap[0][y][x][0];
                flatHeatmap[y * 64 + x] = val;
                if (val > maxConfidence) {
                    maxConfidence = val;
                }
            }
        }

        // Integrity Guard: Check if target is lost due to low confidence peak
        if (maxConfidence < CONFIDENCE_THRESHOLD) {
            handleTargetLost("Low confidence peak (val: " + String.format("%.2f", maxConfidence) + ")");
            return;
        }

        // 5. Execute JNI Center of Mass
        long postStart = SystemClock.elapsedRealtime();
        float[] predCoords = MainActivity.calculateCenterOfMass(flatHeatmap, 0.1f);
        long postDuration = SystemClock.elapsedRealtime() - postStart;

        long totalDuration = SystemClock.elapsedRealtime() - startTime;

        float px = predCoords[0]; 
        float py = predCoords[1]; 

        // Integrity Guard: Check if predicted target position went out-of-bounds
        if (px < 0.0f || px > 1.0f || py < 0.0f || py > 1.0f) {
            handleTargetLost("Target went out of bounds.");
            return;
        }

        // 6. Update UI and Render Outputs
        drawTrackingIndicator(px, py, currentFrame);
        renderDiagnostics(flatHeatmap, histBuffer, currBuffer);

        // Update Telemetry Panel
        txtLatency.setText(String.format("Latency: JNI Pre:%dms | TFLite:%dms | CoM:%dms (Total:%dms)", 
                preDuration, infDuration, postDuration, totalDuration));
        txtPredictedCoords.setText(String.format("Target Position: (%.3f, %.3f) [Confidence: %.2f]", px, py, maxConfidence));
        txtBufferStatus.setText("Tracking Engine: Active");
    }

    private void handleTargetLost(String reason) {
        currentUiState = STATE_LOST;
        
        // Render red overlay HUD message
        tutorialHud.setText("איבדנו את המטרה!");
        tutorialHud.setTextColor(Color.RED);
        lblStatus.setText("Status: Target Lost! (" + reason + ")");
        
        // Draw red outline indicating failure
        Bitmap mutableOverlay = Bitmap.createBitmap(capturedImageView.getWidth(), capturedImageView.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(mutableOverlay);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(12.0f);
        canvas.drawRect(0, 0, canvas.getWidth(), canvas.getHeight(), paint);
        capturedImageView.setImageBitmap(mutableOverlay);
        
        Toast.makeText(this, "Target Lost: " + reason, Toast.LENGTH_LONG).show();
    }

    private void drawTrackingIndicator(float tx, float ty, Bitmap referenceBitmap) {
        int viewW = capturedImageView.getWidth();
        int viewH = capturedImageView.getHeight();
        if (viewW <= 0 || viewH <= 0 || referenceBitmap == null) return;

        Bitmap overlayBitmap = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(overlayBitmap);
        
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(6.0f);
        paint.setAntiAlias(true);
        
        float[] screenCoords = mapFrameCoordsToScreen(tx, ty, referenceBitmap);
        if (screenCoords != null) {
            float sx = screenCoords[0];
            float sy = screenCoords[1];
            
            // Green color for active tracking
            paint.setColor(Color.GREEN);
            canvas.drawCircle(sx, sy, 25.0f, paint);
            
            paint.setStyle(Paint.Style.FILL);
            canvas.drawCircle(sx, sy, 6.0f, paint);
        }
        
        capturedImageView.setImageBitmap(overlayBitmap);
    }

    private void renderDiagnostics(float[] heatmap, float[] hist, float[] curr) {
        // A. Render Heatmap (Jet Colormap)
        Bitmap hmBitmap = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
        int[] hmColors = new int[64 * 64];
        for (int i = 0; i < 64 * 64; i++) {
            float val = heatmap[i];
            val = Math.max(0.0f, Math.min(val, 1.0f));
            int r = (int)(val * 255.0f);
            int b = (int)((1.0f - val) * 255.0f);
            int g = (int)(val * 100.0f);
            hmColors[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        hmBitmap.setPixels(hmColors, 0, 64, 0, 0, 64, 64);
        heatmapImageView.setImageBitmap(hmBitmap);

        // B. Render crops
        cropHistView.setImageBitmap(renderGrayscaleCrop(hist, 2));
        cropCurrView.setImageBitmap(renderGrayscaleCrop(curr, 1));
    }

    private Bitmap renderGrayscaleCrop(float[] floatBuffer, int channels) {
        Bitmap cropBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
        int[] colors = new int[256 * 256];
        for (int i = 0; i < 256 * 256; i++) {
            int grayVal = (int)(floatBuffer[channels * i] * 255.0f);
            grayVal = Math.max(0, Math.min(grayVal, 255));
            
            int r = grayVal;
            int g = grayVal;
            int b = grayVal;
            
            if (channels >= 2) {
                float maskVal = floatBuffer[channels * i + 1];
                if (maskVal > 0.01f) {
                    float alpha = 0.35f * maskVal;
                    r = (int) (grayVal * (1.0f - alpha) + 255.0f * alpha);
                    g = (int) (grayVal * (1.0f - alpha));
                    b = (int) (grayVal * (1.0f - alpha));
                }
            }
            
            colors[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        cropBitmap.setPixels(colors, 0, 256, 0, 0, 256, 256);
        return cropBitmap;
    }

    private float[] getNormalizedCoords(float viewX, float viewY) {
        int viewWidth = capturedImageView.getWidth();
        int viewHeight = capturedImageView.getHeight();
        if (viewWidth <= 0 || viewHeight <= 0) return null;
        return new float[]{ viewX / viewWidth, viewY / viewHeight };
    }

    private float[] mapFrameCoordsToScreen(float px, float py, Bitmap referenceBitmap) {
        if (referenceBitmap == null) return null;
        int viewWidth = capturedImageView.getWidth();
        int viewHeight = capturedImageView.getHeight();
        int imgWidth = referenceBitmap.getWidth();
        int imgHeight = referenceBitmap.getHeight();
        
        float viewRatio = (float) viewWidth / viewHeight;
        float imgRatio = (float) imgWidth / imgHeight;
        
        float screenX, screenY;
        if (imgRatio > viewRatio) { // Fit Width, height is letterboxed
            float actualHeight = viewWidth / imgRatio;
            float offsetY = (viewHeight - actualHeight) / 2.0f;
            screenX = px * viewWidth;
            screenY = offsetY + py * actualHeight;
        } else { // Fit Height, width is letterboxed
            float actualWidth = viewHeight * imgRatio;
            float offsetX = (viewWidth - actualWidth) / 2.0f;
            screenX = offsetX + px * actualWidth;
            screenY = py * viewHeight;
        }
        return new float[]{ screenX, screenY };
    }

    private void resetTrackerToIdle() {
        currentUiState = STATE_IDLE;
        histBitmap = null;
        
        btnReset.setVisibility(View.GONE);
        resultsPanel.setVisibility(View.GONE);
        
        tutorialHud.setText("Tap screen to select target");
        tutorialHud.setTextColor(Color.parseColor("#00e6ff"));
        lblStatus.setText("Status: Camera active. Select target!");
        
        // Clear overlay drawing
        Bitmap emptyBitmap = Bitmap.createBitmap(capturedImageView.getWidth(), capturedImageView.getHeight(), Bitmap.Config.ARGB_8888);
        capturedImageView.setImageBitmap(emptyBitmap);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (cameraHelper.hasCameraPermission()) {
                startCameraStream();
            } else {
                Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        isLoopActive = false;
        trackingHandler.removeCallbacks(trackingRunnable);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (currentUiState == STATE_TRACKING && !isLoopActive) {
            isLoopActive = true;
            trackingHandler.post(trackingRunnable);
        }
    }

    @Override
    protected void onDestroy() {
        isLoopActive = false;
        trackingHandler.removeCallbacks(trackingRunnable);
        if (tflite != null) {
            tflite.close();
        }
        if (cameraHelper != null) {
            cameraHelper.shutdown();
        }
        super.onDestroy();
    }
}
