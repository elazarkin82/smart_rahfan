package elazarkin.ksg.external.trackertester;

import android.Manifest;
import android.content.pm.PackageManager;
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
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

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
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
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

        // 5. Request Camera Permissions
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
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
        Bitmap bitmap = viewFinder.getBitmap();
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

        // 1. Allocate input arrays for TargetTracker2 (flat 256*256 floats)
        float[] histBuffer = new float[256 * 256];
        float[] prevBuffer = new float[256 * 256];
        float[] currBuffer = new float[256 * 256];

        long preStart = SystemClock.elapsedRealtime();
        // 2. Invoke JNI Preprocessing: Crop and mask using identical captured frame
        // - hist_frame: scale 0.40f, circular mask 128px
        MainActivity.cropAndMaskFrame(capturedBitmap, tx, ty, 0.40f, 128.0f, histBuffer);
        // - prev_frame: scale 0.34f, circular mask 50px
        MainActivity.cropAndMaskFrame(capturedBitmap, tx, ty, 0.34f, 50.0f, prevBuffer);
        // - curr_frame: scale 0.28f, unmasked (radius = 0)
        MainActivity.cropAndMaskFrame(capturedBitmap, tx, ty, 0.28f, 0.0f, currBuffer);
        long preDuration = SystemClock.elapsedRealtime() - preStart;

        // 3. Prepare inputs/outputs for TFLite multiple inputs execution
        // Reshape flat float arrays to [1][256][256][1] expected by model
        float[][][][] histInput = new float[1][256][256][1];
        float[][][][] prevInput = new float[1][256][256][1];
        float[][][][] currInput = new float[1][256][256][1];

        for (int i = 0; i < 256 * 256; i++) {
            int y = i / 256;
            int x = i % 256;
            histInput[0][y][x][0] = histBuffer[i];
            prevInput[0][y][x][0] = prevBuffer[i];
            currInput[0][y][x][0] = currBuffer[i];
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

        float dx = predCoords[0]; // predicted X relative to current crop window [0, 1]
        float dy = predCoords[1]; // predicted Y relative to current crop window [0, 1]

        // 7. Translate predicted relative offset back to absolute image coordinates
        // Current crop window scale is 0.28f, centered around (tx, ty)
        float cropScale = 0.28f;
        float x1 = tx - cropScale / 2.0f;
        float y1 = ty - cropScale / 2.0f;
        
        // Match C++ JNI clamping logic for absolute coordinates
        x1 = Math.max(0.0f, Math.min(x1, 1.0f - cropScale));
        y1 = Math.max(0.0f, Math.min(y1, 1.0f - cropScale));
        
        float px = x1 + dx * cropScale;
        float py = y1 + dy * cropScale;

        // 8. Render Results on Screen
        renderOutputs(tx, ty, px, py, flatHeatmap, histBuffer, prevBuffer, currBuffer);

        // 9. Update Telemetry text
        txtLatency.setText(String.format("Latency: JNI Pre:%dms | TFLite:%dms | JNI Post:%dms (Total:%dms)", 
                preDuration, infDuration, postDuration, totalDuration));
        txtSelectedCoords.setText(String.format("Selected Coordinate: (%.3f, %.3f)", tx, ty));
        txtPredictedCoords.setText(String.format("Predicted CoM: (%.3f, %.3f) [dx:%.3f, dy:%.3f]", px, py, dx, dy));
        
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
            int grayVal = (int)(floatBuffer[i] * 255.0f);
            grayVal = Math.max(0, Math.min(grayVal, 255));
            colors[i] = 0xFF000000 | (grayVal << 16) | (grayVal << 8) | grayVal;
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

    private void startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to start CameraX: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
        
        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(this, cameraSelector, preview);
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Camera bind failed: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                startCamera();
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
        super.onDestroy();
    }
}
