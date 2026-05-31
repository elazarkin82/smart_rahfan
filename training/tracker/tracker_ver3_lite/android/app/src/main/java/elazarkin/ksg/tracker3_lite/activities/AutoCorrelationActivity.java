package elazarkin.ksg.tracker3_lite.activities;

import android.Manifest;
import android.content.res.AssetFileDescriptor;
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

public class AutoCorrelationActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1001;

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
    private TextView txtSelectedCoords;
    private TextView txtPredictedCoords;
    private TextView txtOffsetError;
    private ImageView cropHistView;
    private ImageView cropCurrView;

    // Camera & Loop State
    private CameraHelper cameraHelper;
    private boolean isTracking = false;
    private final Handler trackingHandler = new Handler(Looper.getMainLooper());
    private boolean isLoopActive = false;

    // Template Anchor State
    private Bitmap histBitmap = null;
    private float histX = 0.5f;
    private float histY = 0.5f;
    private float[] histBuffer = new float[256 * 256 * 2]; // Grayscale + Mask

    // TFLite State
    private Interpreter tflite;

    // 30 FPS repeating loop for active real-time tracking
    private final Runnable trackingRunnable = new Runnable() {
        @Override
        public void run() {
            if (isLoopActive && isTracking) {
                processLiveFrame();
            }
            if (isLoopActive) {
                trackingHandler.postDelayed(this, 33);
            }
        }
    };

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
        cropCurrView = findViewById(R.id.cropCurrView);

        // 2. Load TFLite Model
        try {
            tflite = new Interpreter(loadModelFile());
            lblStatus.setText("Status: Engine initialized successfully");
        } catch (Exception e) {
            e.printStackTrace();
            lblStatus.setText("Status: Failed to load TFLite model!");
            Toast.makeText(this, "TFLite Engine Load Failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }

        // 3. Button Clicks
        btnBack.setOnClickListener(v -> finish());
        btnReset.setOnClickListener(v -> resetTracker());

        // 4. Set Up Touch Interactions on the ViewFinder
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

    private void startCameraStream() {
        lblStatus.setText("Status: Connecting camera...");
        cameraHelper.startCamera(new CameraHelper.OnCameraReadyCallback() {
            @Override
            public void onCameraReady() {
                lblStatus.setText("Status: Active. Select target to lock!");
                isLoopActive = true;
                trackingHandler.post(trackingRunnable);
            }

            @Override
            public void onCameraError(Exception e) {
                lblStatus.setText("Status: Camera bind failed!");
            }
        });
    }

    private void setupTouchInteractions() {
        // Tap on ViewFinder to Lock Target & Continue Stream
        viewFinder.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                float viewX = event.getX();
                float viewY = event.getY();
                
                Bitmap currentFrame = cameraHelper.captureFrame();
                if (currentFrame != null) {
                    int viewWidth = viewFinder.getWidth();
                    int viewHeight = viewFinder.getHeight();
                    float[] imageCoords = MainActivity.mapScreenCoordsToFrame(
                            viewX, viewY, viewWidth, viewHeight, currentFrame.getWidth(), currentFrame.getHeight()
                    );
                    if (imageCoords != null) {
                        lockTargetAt(imageCoords[0], imageCoords[1], currentFrame);
                        return true;
                    }
                }
            }
            return false;
        });
    }

    private void lockTargetAt(float tx, float ty, Bitmap frame) {
        histBitmap = frame;
        histX = tx;
        histY = ty;
        
        // 1. Invoke JNI Preprocessing for template: Grayscale + Exponential Cone Mask (2 channels)
        // - targetX, targetY = tx, ty
        // - maskRadius = 128.0f
        // - useExponentialMask = true
        // - maskSigma = 30.0f
        // - isSearchFrame = false
        // - numChannels = 2
        MainActivity.downsampleAndMaskFrameV3(histBitmap, histX, histY, 128.0f, true, 30.0f, false, 2, histBuffer);
        
        isTracking = true;
        btnReset.setVisibility(View.VISIBLE);
        resultsPanel.setVisibility(View.VISIBLE);
        tutorialHud.setText("Target locked! Tap again to lock another target");
        lblStatus.setText("Status: Target locked. Tracking live...");
    }

    private void processLiveFrame() {
        if (tflite == null || histBitmap == null) return;

        Bitmap currBitmap = cameraHelper.captureFrame();
        if (currBitmap == null) return;

        long startTime = SystemClock.elapsedRealtime();

        // 1. Preprocess search frame: Grayscale only (1 channel, no mask)
        float[] currBuffer = new float[256 * 256 * 1];
        long preStart = SystemClock.elapsedRealtime();
        MainActivity.downsampleAndMaskFrameV3(currBitmap, 0.0f, 0.0f, 0.0f, false, 0.0f, true, 1, currBuffer);
        long preDuration = SystemClock.elapsedRealtime() - preStart;

        // 2. Format inputs/outputs for TFLite
        float[][][][] histInput = new float[1][256][256][2];
        float[][][][] currInput = new float[1][256][256][1];

        // Fill histInput (2 channels)
        for (int i = 0; i < 256 * 256; i++) {
            int y = i / 256;
            int x = i % 256;
            histInput[0][y][x][0] = histBuffer[2 * i];
            histInput[0][y][x][1] = histBuffer[2 * i + 1];
        }

        // Fill currInput (1 channel)
        for (int i = 0; i < 256 * 256; i++) {
            int y = i / 256;
            int x = i % 256;
            currInput[0][y][x][0] = currBuffer[i];
        }

        Object[] inputs = new Object[]{ histInput, currInput };
        float[][][][] outputHeatmap = new float[1][64][64][1];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputHeatmap);

        // 3. Execute inference
        long infStart = SystemClock.elapsedRealtime();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long infDuration = SystemClock.elapsedRealtime() - infStart;

        // 4. Flatten the 64x64 output heatmap for postprocessing
        float[] flatHeatmap = new float[64 * 64];
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                flatHeatmap[y * 64 + x] = outputHeatmap[0][y][x][0];
            }
        }

        // 5. Invoke JNI Center of Mass (threshold = 0.1f)
        long postStart = SystemClock.elapsedRealtime();
        float[] predCoords = MainActivity.calculateCenterOfMass(flatHeatmap, 0.1f);
        long postDuration = SystemClock.elapsedRealtime() - postStart;

        long totalDuration = SystemClock.elapsedRealtime() - startTime;

        float px = predCoords[0];
        float py = predCoords[1];

        // 6. Draw Indicators on overlay canvas
        renderIndicators(histX, histY, px, py, flatHeatmap, histBuffer, currBuffer, currBitmap);

        // 7. Update Telemetry text
        txtLatency.setText(String.format("Latency: JNI Pre:%dms | TFLite:%dms | CoM:%dms (Total:%dms)", 
                preDuration, infDuration, postDuration, totalDuration));
        txtSelectedCoords.setText(String.format("Original Anchor: (%.3f, %.3f)", histX, histY));
        txtPredictedCoords.setText(String.format("Current Target CoM: (%.3f, %.3f)", px, py));
        
        // Calculate error in pixels relative to current preview bitmap
        float errX = (px - histX) * currBitmap.getWidth();
        float errY = (py - histY) * currBitmap.getHeight();
        double errorPx = Math.sqrt(errX * errX + errY * errY);
        txtOffsetError.setText(String.format("Displacement: %.2f pixels", errorPx));
    }

    private void renderIndicators(float tx, float ty, float px, float py, 
                                  float[] heatmap, float[] hist, float[] curr, Bitmap currentFrame) {
        
        int viewW = capturedImageView.getWidth();
        int viewH = capturedImageView.getHeight();
        if (viewW <= 0 || viewH <= 0 || currentFrame == null) return;

        // Create transparent mutable drawing canvas matching views
        Bitmap overlayBitmap = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(overlayBitmap);
        
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5.0f);
        paint.setAntiAlias(true);
        
        // A. Draw original anchor target (Yellow)
        float[] screenHist = mapFrameCoordsToScreen(tx, ty, currentFrame);
        if (screenHist != null) {
            paint.setColor(Color.YELLOW);
            canvas.drawCircle(screenHist[0], screenHist[1], 20.0f, paint);
            paint.setStyle(Paint.Style.FILL);
            canvas.drawCircle(screenHist[0], screenHist[1], 5.0f, paint);
        }
        
        // B. Draw predicted target (Neon Green)
        float[] screenPred = mapFrameCoordsToScreen(px, py, currentFrame);
        if (screenPred != null) {
            paint.setStyle(Paint.Style.STROKE);
            paint.setColor(Color.GREEN);
            canvas.drawCircle(screenPred[0], screenPred[1], 20.0f, paint);
            paint.setStyle(Paint.Style.FILL);
            canvas.drawCircle(screenPred[0], screenPred[1], 5.0f, paint);
        }
        
        capturedImageView.setImageBitmap(overlayBitmap);

        // C. Render predicted Heatmap (Colormap: Jet approximation)
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

        // D. Render Visual Debug Crops (256x256)
        cropHistView.setImageBitmap(renderGrayscaleCrop(hist, 2));
        cropCurrView.setImageBitmap(renderGrayscaleCrop(curr, 1));
    }

    private Bitmap renderGrayscaleCrop(float[] floatBuffer, int channels) {
        Bitmap cropBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
        int[] colors = new int[256 * 256];
        for (int i = 0; i < 256 * 256; i++) {
            // Channel 0 is always grayscale
            int grayVal = (int)(floatBuffer[channels * i] * 255.0f);
            grayVal = Math.max(0, Math.min(grayVal, 255));
            
            int r = grayVal;
            int g = grayVal;
            int b = grayVal;
            
            // Channel 1: blend glow red attention mask if present
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
        int viewWidth = viewFinder.getWidth();
        int viewHeight = viewFinder.getHeight();
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

    private void resetTracker() {
        isTracking = false;
        histBitmap = null;
        
        btnReset.setVisibility(View.GONE);
        resultsPanel.setVisibility(View.GONE);
        
        tutorialHud.setText("Tap screen to select target");
        lblStatus.setText("Status: Active. Select target to lock!");
        
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
        if (cameraHelper.hasCameraPermission() && !isLoopActive) {
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
