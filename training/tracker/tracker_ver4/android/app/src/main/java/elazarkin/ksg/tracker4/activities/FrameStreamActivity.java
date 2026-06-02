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
import android.widget.SeekBar;

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
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;

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
    private Button btnBack;

    private LinearLayout resultsPanel;
    private ImageView heatmapImageView;
    private TextView txtLatency;
    private TextView txtPredictedCoords;
    private TextView txtBufferStatus;
    private ImageView cropHistView;
    private ImageView cropCurrView;
    private ImageView cropFullView;
    
    private TextView lblHistStack;
    private SeekBar seekBarHistLayer;
    private int selectedHistLayer = 0;

    private float[] cachedFlatHeatmap = null;
    private float[] cachedCurrBuffer = null;
    private Bitmap cachedFullFrameBmp = null;
    private float cachedPx = 0.5f;
    private float cachedPy = 0.5f;

    private CameraHelper cameraHelper;
    private boolean isLoopActive = false;
    
    private float targetX = 0.0f;
    private float targetY = 0.0f;
    
    // Ver4 Inputs: 1 channel grayscale
    private float[][][][][] refStackInput = new float[1][16][32][32][1];

    private Interpreter tflite;

    // Asynchronous Producer-Consumer Threading & Locking Fields
    private final ReentrantLock frameLock = new ReentrantLock();
    private final Condition frameCondition = frameLock.newCondition();
    private byte[] sharedFrameData = null;
    private byte[] rawYPlaneBuffer = null;
    private int sharedFrameW = 0;
    private int sharedFrameH = 0;
    private int sharedFrameStride = 0;
    private int sharedFrameRotation = 0;
    private boolean hasSharedFrame = false;

    private Thread workerThread = null;
    private volatile boolean isWorkerActive = false;

    // Tracking position state in absolute camera frame pixels
    private float lastTrackedX = 0.0f;
    private float lastTrackedY = 0.0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_frame_stream);

        viewFinder = findViewById(R.id.viewFinder);
        capturedImageView = findViewById(R.id.capturedImageView);
        tutorialHud = findViewById(R.id.tutorial_hud);
        lblStatus = findViewById(R.id.lbl_status);
        btnBack = findViewById(R.id.btn_back);

        resultsPanel = findViewById(R.id.results_panel);
        heatmapImageView = findViewById(R.id.heatmapImageView);
        txtLatency = findViewById(R.id.txt_latency);
        txtPredictedCoords = findViewById(R.id.txt_predicted_coords);
        txtBufferStatus = findViewById(R.id.txt_buffer_status);
        cropHistView = findViewById(R.id.cropHistView);
        cropCurrView = findViewById(R.id.cropCurrView);
        cropFullView = findViewById(R.id.cropFullView);

        lblHistStack = findViewById(R.id.lblHistStack);
        seekBarHistLayer = findViewById(R.id.seekBarHistLayer);
        seekBarHistLayer.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                selectedHistLayer = progress;
                lblHistStack.setText("Hist Stack (L: " + selectedHistLayer + ")");
                if (cachedFlatHeatmap != null && cachedCurrBuffer != null) {
                    renderDiagnostics(cachedFlatHeatmap, cachedCurrBuffer, cachedFullFrameBmp, cachedPx, cachedPy);
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            tflite = new Interpreter(loadModelFile(), options);
            lblStatus.setText("Status: Engine loaded successfully");
        } catch (Exception e) {
            e.printStackTrace();
            lblStatus.setText("Status: Failed to load TFLite model!");
        }

        btnBack.setOnClickListener(v -> finish());
        
        capturedImageView.setOnTouchListener((v, event) -> {
            if (event.getAction() == android.view.MotionEvent.ACTION_DOWN) {
                targetX = event.getX();
                targetY = event.getY();
                currentUiState = STATE_GATHERING;
                lblStatus.setText("Status: Gathering Reference Frames...");
                tutorialHud.setText("Hold steady on target...");
                tutorialHud.setTextColor(Color.YELLOW);
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

        int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();

        androidx.camera.core.ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        java.nio.ByteBuffer yBuffer = planes[0].getBuffer();
        int yRowStride = planes[0].getRowStride();
        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        yBuffer.rewind();
        int length = yBuffer.remaining();
        if (rawYPlaneBuffer == null || rawYPlaneBuffer.length != length) {
            rawYPlaneBuffer = new byte[length];
        }
        yBuffer.get(rawYPlaneBuffer);

        int rotW = (rotationDegrees == 90 || rotationDegrees == 270) ? height : width;
        int rotH = (rotationDegrees == 90 || rotationDegrees == 270) ? width : height;
        int rotLength = rotW * rotH;

        if (currentUiState == STATE_GATHERING) {
            byte[] rotatedData = new byte[rotLength];
            MainActivity.rotateYPlane(rawYPlaneBuffer, rotatedData, width, height, yRowStride, rotationDegrees);
            gatherReferenceFrame(rotatedData, rotW, rotH);
            imageProxy.close();
        } else if (currentUiState == STATE_TRACKING) {
            // Producer: Lock, write rotated Y-plane bytes to shared buffer, and signal the Worker Thread
            frameLock.lock();
            try {
                if (sharedFrameData == null || sharedFrameData.length != rotLength) {
                    sharedFrameData = new byte[rotLength];
                }
                MainActivity.rotateYPlane(rawYPlaneBuffer, sharedFrameData, width, height, yRowStride, rotationDegrees);
                sharedFrameW = rotW;
                sharedFrameH = rotH;
                sharedFrameStride = rotW;
                sharedFrameRotation = 0; // Already rotated to 0 degrees
                hasSharedFrame = true;
                frameCondition.signal();
            } finally {
                frameLock.unlock();
            }
            imageProxy.close();
        }
    }

    private void gatherReferenceFrame(byte[] yPlane, int width, int height) {
        float[] normCoords = MainActivity.mapAlignedScreenCoordsToFrame(targetX, targetY, viewFinder.getWidth(), viewFinder.getHeight(), width, height);
        if (normCoords == null) {
            currentUiState = STATE_IDLE;
            return;
        }
        
        float cx = normCoords[0] * width;
        float cy = normCoords[1] * height;

        // Initialize tracking position state
        lastTrackedX = cx;
        lastTrackedY = cy;
        
        // Zoom range matches pipeline_config.json: from size 128 down to 4 pixels (scaled dynamically by camera height relative to 600px training size)
        float maxCropSize = (128.0f / 600.0f) * height;
        float minCropSize = (4.0f / 600.0f) * height;
        
        int stride = width; // rotated frame stride is width
        
        for (int layer = 0; layer < 16; layer++) {
            float size = maxCropSize - layer * ((maxCropSize - minCropSize) / 15.0f);
            float half = size / 2.0f;
            
            for (int y = 0; y < 32; y++) {
                for (int x = 0; x < 32; x++) {
                    float srcX = cx - half + (x / 31.0f) * size;
                    float srcY = cy - half + (y / 31.0f) * size;
                    
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

    private void startWorkerThread() {
        if (workerThread != null) return;
        isWorkerActive = true;
        workerThread = new Thread(new Runnable() {
            @Override
            public void run() {
                workerLoop();
            }
        }, "TrackerWorkerThread");
        workerThread.start();
    }

    private void stopWorkerThread() {
        isWorkerActive = false;
        if (workerThread != null) {
            frameLock.lock();
            try {
                frameCondition.signalAll();
            } finally {
                frameLock.unlock();
            }
            try {
                workerThread.join(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            workerThread = null;
        }
    }

    private void workerLoop() {
        byte[] localFrameData = null;
        int localW = 0;
        int localH = 0;
        int localStride = 0;
        int localRotation = 0;

        while (isWorkerActive) {
            frameLock.lock();
            try {
                while (isWorkerActive && !hasSharedFrame) {
                    frameCondition.await();
                }
                if (!isWorkerActive) return;

                // Swap/copy frame bytes under lock to local buffers
                if (sharedFrameData != null) {
                    if (localFrameData == null || localFrameData.length != sharedFrameData.length) {
                        localFrameData = new byte[sharedFrameData.length];
                    }
                    System.arraycopy(sharedFrameData, 0, localFrameData, 0, sharedFrameData.length);
                    localW = sharedFrameW;
                    localH = sharedFrameH;
                    localStride = sharedFrameStride;
                    localRotation = sharedFrameRotation;
                }
                hasSharedFrame = false;
            } catch (InterruptedException e) {
                return;
            } finally {
                frameLock.unlock();
            }

            if (localFrameData != null) {
                // Execute heavy inference and math operations on background thread outside lock
                processWorkerFrame(localFrameData, localW, localH, localStride, localRotation);
            }
        }
    }

    private void processWorkerFrame(byte[] yPlane, int width, int height, int stride, int rotationDegrees) {
        if (tflite == null) return;

        long startTime = SystemClock.elapsedRealtime();

        // Calculate search crop size: min(width, height)
        float cropSize = (float) Math.min(width, height);
        
        float[] currBuffer = new float[256 * 256];
        long preStart = SystemClock.elapsedRealtime();
        
        // JNI extracts square crop around lastTrackedX, lastTrackedY, resizes to 256x256, and normalizes
        MainActivity.downsampleSearchCrop(yPlane, width, height, stride, lastTrackedX, lastTrackedY, cropSize, currBuffer);
        
        long preDuration = SystemClock.elapsedRealtime() - preStart;

        float[][][][] currInput = new float[1][256][256][1];
        for (int i = 0; i < 256 * 256; i++) {
            int y = i / 256;
            int x = i % 256;
            currInput[0][y][x][0] = currBuffer[i];
        }

        Object[] inputs = new Object[]{ refStackInput, currInput };
        float[][][][] outputHeatmap = new float[1][256][256][1];
        float[][] outputQuality = new float[1][1];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputHeatmap);
        outputs.put(1, outputQuality);

        long infStart = SystemClock.elapsedRealtime();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long infDuration = SystemClock.elapsedRealtime() - infStart;

        float qualityScore = outputQuality[0][0];
        
        // Flatten predicted heatmap for JNI processing
        float[] flatHeatmap = new float[256 * 256];
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                flatHeatmap[y * 256 + x] = outputHeatmap[0][y][x][0];
            }
        }

        long postStart = SystemClock.elapsedRealtime();
        // Calculate noise-immune sub-pixel centroid
        float[] localCoords = MainActivity.calculateLocalRefinedArgmaxCentroid(flatHeatmap);
        long postDuration = SystemClock.elapsedRealtime() - postStart;
        
        float px = localCoords[0]; // relative x in [0.0, 1.0] inside the crop
        float py = localCoords[1]; // relative y in [0.0, 1.0] inside the crop
        
        // Coordinate Re-projection: Map crop-relative coordinates back to camera absolute coordinates
        float halfSize = cropSize / 2.0f;
        float x_global = (lastTrackedX - halfSize) + px * cropSize;
        float y_global = (lastTrackedY - halfSize) + py * cropSize;

        // Clamp target position to camera frame boundaries to keep tracking running continuously
        x_global = Math.max(0.0f, Math.min((float)width - 1.0f, x_global));
        y_global = Math.max(0.0f, Math.min((float)height - 1.0f, y_global));

        // Save for next frame centering
        lastTrackedX = x_global;
        lastTrackedY = y_global;

        long totalDuration = SystemClock.elapsedRealtime() - startTime;

        // Convert absolute camera coordinates to normalized [0, 1] relative to camera frame
        float gx = x_global / (float) width;
        float gy = y_global / (float) height;

        // Since the frame is rotated to screen space, screen normalization is direct (1:1)
        final float screenX_norm = gx;
        final float screenY_norm = gy;

        // Downsample the full processed frame to a small bitmap to inspect rotation/orientation
        Bitmap tmpBmp = null;
        try {
            int downsampleFactor = 8;
            int smallW = width / downsampleFactor;
            int smallH = height / downsampleFactor;
            tmpBmp = Bitmap.createBitmap(smallW, smallH, Bitmap.Config.ARGB_8888);
            int[] pixels = new int[smallW * smallH];
            for (int y = 0; y < smallH; y++) {
                int srcY = y * downsampleFactor;
                int rowOffset = srcY * stride;
                for (int x = 0; x < smallW; x++) {
                    int srcX = x * downsampleFactor;
                    int val = yPlane[rowOffset + srcX] & 0xFF;
                    pixels[y * smallW + x] = 0xFF000000 | (val << 16) | (val << 8) | val;
                }
            }
            tmpBmp.setPixels(pixels, 0, smallW, 0, 0, smallW, smallH);
            
            // Draw a small indicator on the downsampled frame showing the tracking target position
            Canvas canvas = new Canvas(tmpBmp);
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.FILL);
            paint.setAntiAlias(true);
            float scaledX = lastTrackedX / (float) downsampleFactor;
            float scaledY = lastTrackedY / (float) downsampleFactor;
            canvas.drawCircle(scaledX, scaledY, 3.0f, paint);
        } catch (Exception e) {
            e.printStackTrace();
        }
        final Bitmap fullFrameBmp = tmpBmp;

        // Save to cache for SeekBar scrolling redraws
        cachedFlatHeatmap = flatHeatmap;
        cachedCurrBuffer = currBuffer;
        cachedFullFrameBmp = fullFrameBmp;
        cachedPx = px;
        cachedPy = py;

        // Post UI rendering tasks back to the UI thread
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                if (currentUiState != STATE_TRACKING) return;
                
                // If quality is below threshold, color the target circle in RED. If above, GREEN.
                int circleColor = (qualityScore >= CONFIDENCE_THRESHOLD) ? Color.GREEN : Color.RED;
                
                if (qualityScore < CONFIDENCE_THRESHOLD) {
                    lblStatus.setText(String.format("Status: Weak Lock! Quality: %.2f", qualityScore));
                } else {
                    lblStatus.setText("Status: Active tracking");
                }
                
                drawTrackingIndicator(lastTrackedX, lastTrackedY, width, height, circleColor);
                renderDiagnostics(flatHeatmap, currBuffer, fullFrameBmp, px, py);

                txtLatency.setText(String.format("Latency: Pre:%dms | TFLite:%dms | CoM:%dms (Total:%dms)", 
                        preDuration, infDuration, postDuration, totalDuration));
                txtPredictedCoords.setText(String.format("Target Pos: (%.3f, %.3f) [Quality: %.2f]", screenX_norm, screenY_norm, qualityScore));
                txtBufferStatus.setText("Tracking Engine: Active");
            }
        });
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

    private void drawTrackingIndicator(float cx, float cy, int imgW, int imgH, int color) {
        if (imgW <= 0 || imgH <= 0) return;

        Bitmap overlayBitmap = Bitmap.createBitmap(imgW, imgH, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(overlayBitmap);
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        
        // Scale circle indicator sizes relative to the camera frame width
        float strokeW = (6.0f / 720.0f) * imgW;
        float circleR = (25.0f / 720.0f) * imgW;
        float innerR = (6.0f / 720.0f) * imgW;
        
        paint.setStrokeWidth(strokeW);
        paint.setAntiAlias(true);
        
        paint.setColor(color);
        canvas.drawCircle(cx, cy, circleR, paint);
        paint.setStyle(Paint.Style.FILL);
        canvas.drawCircle(cx, cy, innerR, paint);
        
        capturedImageView.setImageBitmap(overlayBitmap);
    }

    private void renderDiagnostics(float[] heatmap, float[] curr, Bitmap fullFrameBmp, float px, float py) {
        // 1. Render outputHeatmap (Jet color mapping)
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

        // 2. Render cropCurrView (the current active search crop)
        Bitmap currBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
        int[] currColors = new int[256 * 256];
        for (int i = 0; i < 256 * 256; i++) {
            int val = (int)(curr[i] * 255.0f);
            currColors[i] = 0xFF000000 | (val << 16) | (val << 8) | val;
        }
        currBitmap.setPixels(currColors, 0, 256, 0, 0, 256, 256);
        
        // Draw a small red indicator showing the predicted target position inside search crop
        Canvas canvasCurr = new Canvas(currBitmap);
        Paint paintCurr = new Paint();
        paintCurr.setColor(Color.RED);
        paintCurr.setStyle(Paint.Style.FILL);
        paintCurr.setAntiAlias(true);
        canvasCurr.drawCircle(px * 256.0f, py * 256.0f, 6.0f, paintCurr);
        
        cropCurrView.setImageBitmap(currBitmap);

        // 3. Render cropHistView (the locked target reference template layer)
        Bitmap histBitmap = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888);
        int[] histColors = new int[32 * 32];
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int val = (int)(refStackInput[0][selectedHistLayer][y][x][0] * 255.0f);
                histColors[y * 32 + x] = 0xFF000000 | (val << 16) | (val << 8) | val;
            }
        }
        histBitmap.setPixels(histColors, 0, 32, 0, 0, 32, 32);
        
        // Draw a small red indicator at the center of the reference crop template (the target origin)
        Canvas canvasHist = new Canvas(histBitmap);
        Paint paintHist = new Paint();
        paintHist.setColor(Color.RED);
        paintHist.setStyle(Paint.Style.FILL);
        paintHist.setAntiAlias(true);
        canvasHist.drawCircle(16.0f, 16.0f, 1.5f, paintHist);
        
        cropHistView.setImageBitmap(histBitmap);

        // 4. Render cropFullView (the downsampled full processed frame)
        if (fullFrameBmp != null) {
            cropFullView.setImageBitmap(fullFrameBmp);
        }
    }

    private void resetTrackerToIdle() {
        currentUiState = STATE_IDLE;
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
        stopWorkerThread();
        super.onPause();
        isLoopActive = false;
    }

    @Override
    protected void onResume() {
        super.onResume();
        startWorkerThread();
        if (cameraHelper != null && cameraHelper.hasCameraPermission() && !isLoopActive) {
            isLoopActive = true;
        }
    }

    @Override
    protected void onDestroy() {
        isLoopActive = false;
        stopWorkerThread();
        if (tflite != null) {
            tflite.close();
        }
        if (cameraHelper != null) {
            cameraHelper.shutdown();
        }
        super.onDestroy();
    }
}
