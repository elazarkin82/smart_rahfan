package elazarkin.ksg.tracker4.activities;

import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
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

import elazarkin.ksg.tracker4.MainActivity;
import elazarkin.ksg.tracker4.R;
import elazarkin.ksg.tracker4.base.camera.CameraHelper;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class FrameStreamActivity extends AppCompatActivity implements CameraHelper.FrameProcessor {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1002;
    private static final float CONFIDENCE_THRESHOLD = 0.20f;

    private static final int STATE_IDLE = 0;
    private static final int STATE_GATHERING = 1;
    private static final int STATE_TRACKING = 2;
    private static final int STATE_LOST = 3;

    private int currentUiState = STATE_IDLE;

    private PreviewView viewFinder;
    private ImageView capturedImageView;
    private TextView tutorialHud;
    private TextView lblStatus;
    private Button btnReset;
    private Button btnBack;

    private LinearLayout resultsPanel;
    private ImageView heatmapImageView;
    private TextView txtLatency;
    private TextView txtPredictedCoords;
    private TextView txtBufferStatus;
    private ImageView cropHistView;
    private ImageView cropCurrView;

    private CameraHelper cameraHelper;
    private boolean isLoopActive = false;
    
    private float targetX = 0.0f;
    private float targetY = 0.0f;
    
    // Ver4 Inputs: 1 channel grayscale
    private float[][][][][] refStackInput = new float[1][16][16][16][1];

    private Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_frame_stream);

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

        try {
            tflite = new Interpreter(loadModelFile());
            lblStatus.setText("Status: Engine loaded successfully");
        } catch (Exception e) {
            e.printStackTrace();
            lblStatus.setText("Status: Failed to load TFLite model!");
        }

        btnBack.setOnClickListener(v -> finish());
        btnReset.setOnClickListener(v -> resetTrackerToIdle());
        
        capturedImageView.setOnTouchListener((v, event) -> {
            if (event.getAction() == android.view.MotionEvent.ACTION_DOWN) {
                if (currentUiState == STATE_IDLE) {
                    targetX = event.getX();
                    targetY = event.getY();
                    currentUiState = STATE_GATHERING;
                    btnReset.setVisibility(View.VISIBLE);
                    lblStatus.setText("Status: Gathering Reference Frames...");
                    tutorialHud.setText("Hold steady on target...");
                    tutorialHud.setTextColor(Color.YELLOW);
                }
            }
            return true;
        });

        cameraHelper = new CameraHelper(this, viewFinder);
        cameraHelper.setFrameProcessor(this);
        if (cameraHelper.hasCameraPermission()) {
            startCameraStream();
        } else {
            cameraHelper.requestCameraPermission(this, CAMERA_PERMISSION_REQUEST_CODE);
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        try {
            AssetFileDescriptor fileDescriptor = this.getAssets().openFd("tracker.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            throw new RuntimeException("tracker.tflite not found in assets");
        }
    }

    private void startCameraStream() {
        lblStatus.setText("Status: Connecting camera...");
        cameraHelper.startCamera(new CameraHelper.OnCameraReadyCallback() {
            @Override
            public void onCameraReady() {
                resetTrackerToIdle();
                isLoopActive = true;
            }
            @Override
            public void onCameraError(Exception e) {
                lblStatus.setText("Status: Camera bind failed!");
            }
        });
    }

    @Override
    public void process(androidx.camera.core.ImageProxy imageProxy) {
        if (!isLoopActive || (currentUiState != STATE_GATHERING && currentUiState != STATE_TRACKING)) {
            imageProxy.close();
            return;
        }

        androidx.camera.core.ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        java.nio.ByteBuffer yBuffer = planes[0].getBuffer();
        int yRowStride = planes[0].getRowStride();
        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        byte[] yPlaneData = new byte[yBuffer.remaining()];
        yBuffer.get(yPlaneData);

        if (currentUiState == STATE_GATHERING) {
            gatherReferenceFrame(yPlaneData, width, height, yRowStride);
        } else if (currentUiState == STATE_TRACKING) {
            processNextFrame(yPlaneData, width, height, yRowStride);
        }
        
        imageProxy.close();
    }

    private void gatherReferenceFrame(byte[] yPlane, int width, int height, int stride) {
        float[] normCoords = MainActivity.mapScreenCoordsToFrame(targetX, targetY, viewFinder.getWidth(), viewFinder.getHeight(), width, height);
        if (normCoords == null) {
            currentUiState = STATE_IDLE;
            return;
        }
        
        float cx = normCoords[0] * width;
        float cy = normCoords[1] * height;
        
        for (int layer = 0; layer < 16; layer++) {
            float size = 128.0f - layer * ((128.0f - 16.0f) / 15.0f);
            float half = size / 2.0f;
            
            for (int y = 0; y < 16; y++) {
                for (int x = 0; x < 16; x++) {
                    float srcX = cx - half + (x / 15.0f) * size;
                    float srcY = cy - half + (y / 15.0f) * size;
                    
                    int ix = (int) Math.max(0, Math.min(width - 1, srcX));
                    int iy = (int) Math.max(0, Math.min(height - 1, srcY));
                    
                    int pixel = yPlane[iy * stride + ix] & 0xFF;
                    refStackInput[0][layer][y][x][0] = pixel / 255.0f;
                }
            }
        }
        
        currentUiState = STATE_TRACKING;
        
        resultsPanel.setVisibility(View.VISIBLE);
        tutorialHud.setText("Active tracking loop running at 30 FPS");
        tutorialHud.setTextColor(Color.parseColor("#00e6ff"));
        lblStatus.setText("Status: Active tracking");
    }

    private void processNextFrame(byte[] yPlane, int width, int height, int stride) {
        if (tflite == null) return;

        long startTime = SystemClock.elapsedRealtime();

        float[] currBuffer = new float[256 * 256];
        long preStart = SystemClock.elapsedRealtime();
        MainActivity.downsampleSearchFrame(yPlane, width, height, stride, currBuffer);
        long preDuration = SystemClock.elapsedRealtime() - preStart;

        float[][][][] currInput = new float[1][256][256][1];
        for (int i = 0; i < 256 * 256; i++) {
            int y = i / 256;
            int x = i % 256;
            currInput[0][y][x][0] = currBuffer[i];
        }

        Object[] inputs = new Object[]{ refStackInput, currInput };
        float[][][][] outputHeatmap = new float[1][256][256][1];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputHeatmap);

        long infStart = SystemClock.elapsedRealtime();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long infDuration = SystemClock.elapsedRealtime() - infStart;

        float maxConfidence = 0.0f;
        float[] flatHeatmap = new float[256 * 256];
        for (int y = 0; y < 256; y++) {
            for (int x = 0; x < 256; x++) {
                float val = outputHeatmap[0][y][x][0];
                flatHeatmap[y * 256 + x] = val;
                if (val > maxConfidence) {
                    maxConfidence = val;
                }
            }
        }

        if (maxConfidence < CONFIDENCE_THRESHOLD) {
            handleTargetLost("Low confidence peak: " + String.format("%.2f", maxConfidence));
            return;
        }

        long postStart = SystemClock.elapsedRealtime();
        float[] predCoords = MainActivity.calculateCenterOfMass(flatHeatmap, 0.1f);
        long postDuration = SystemClock.elapsedRealtime() - postStart;
        long totalDuration = SystemClock.elapsedRealtime() - startTime;

        float px = predCoords[0]; 
        float py = predCoords[1]; 

        if (px < 0.0f || px > 1.0f || py < 0.0f || py > 1.0f) {
            handleTargetLost("Target went out of bounds.");
            return;
        }

        drawTrackingIndicator(px, py);
        renderDiagnostics(flatHeatmap, currBuffer);

        txtLatency.setText(String.format("Latency: Pre:%dms | TFLite:%dms | CoM:%dms (Total:%dms)", 
                preDuration, infDuration, postDuration, totalDuration));
        txtPredictedCoords.setText(String.format("Target Pos: (%.3f, %.3f) [Conf: %.2f]", px, py, maxConfidence));
        txtBufferStatus.setText("Tracking Engine: Active");
    }

    private void handleTargetLost(String reason) {
        currentUiState = STATE_LOST;
        tutorialHud.setText("Target Lost!");
        tutorialHud.setTextColor(Color.RED);
        lblStatus.setText("Status: Target Lost! (" + reason + ")");
        
        Bitmap mutableOverlay = Bitmap.createBitmap(capturedImageView.getWidth(), capturedImageView.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(mutableOverlay);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(12.0f);
        canvas.drawRect(0, 0, canvas.getWidth(), canvas.getHeight(), paint);
        capturedImageView.setImageBitmap(mutableOverlay);
        
        Toast.makeText(this, "Target Lost: " + reason, Toast.LENGTH_SHORT).show();
    }

    private void drawTrackingIndicator(float tx, float ty) {
        int viewW = capturedImageView.getWidth();
        int viewH = capturedImageView.getHeight();
        if (viewW <= 0 || viewH <= 0) return;

        Bitmap overlayBitmap = Bitmap.createBitmap(viewW, viewH, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(overlayBitmap);
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(6.0f);
        paint.setAntiAlias(true);
        
        float screenX = tx * viewW;
        float screenY = ty * viewH;
        
        paint.setColor(Color.GREEN);
        canvas.drawCircle(screenX, screenY, 25.0f, paint);
        paint.setStyle(Paint.Style.FILL);
        canvas.drawCircle(screenX, screenY, 6.0f, paint);
        
        capturedImageView.setImageBitmap(overlayBitmap);
    }

    private void renderDiagnostics(float[] heatmap, float[] curr) {
        Bitmap hmBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
        int[] hmColors = new int[256 * 256];
        for (int i = 0; i < 256 * 256; i++) {
            float val = Math.max(0.0f, Math.min(heatmap[i], 1.0f));
            int r = (int)(val * 255.0f);
            int b = (int)((1.0f - val) * 255.0f);
            int g = (int)(val * 100.0f);
            hmColors[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        hmBitmap.setPixels(hmColors, 0, 256, 0, 0, 256, 256);
        heatmapImageView.setImageBitmap(hmBitmap);

        Bitmap currBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
        int[] currColors = new int[256 * 256];
        for (int i = 0; i < 256 * 256; i++) {
            int val = (int)(curr[i] * 255.0f);
            currColors[i] = 0xFF000000 | (val << 16) | (val << 8) | val;
        }
        currBitmap.setPixels(currColors, 0, 256, 0, 0, 256, 256);
        cropCurrView.setImageBitmap(currBitmap);
    }

    private void resetTrackerToIdle() {
        currentUiState = STATE_IDLE;
        btnReset.setVisibility(View.GONE);
        resultsPanel.setVisibility(View.GONE);
        
        tutorialHud.setText("Tap screen to lock on target");
        tutorialHud.setTextColor(Color.parseColor("#00e6ff"));
        lblStatus.setText("Status: Ready to lock");
        
        Bitmap emptyBitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888);
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
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (cameraHelper != null && cameraHelper.hasCameraPermission() && !isLoopActive) {
            isLoopActive = true;
        }
    }

    @Override
    protected void onDestroy() {
        isLoopActive = false;
        if (tflite != null) {
            tflite.close();
        }
        if (cameraHelper != null) {
            cameraHelper.shutdown();
        }
        super.onDestroy();
    }
}
