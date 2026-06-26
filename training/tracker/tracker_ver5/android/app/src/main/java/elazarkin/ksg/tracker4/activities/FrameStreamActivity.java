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
import android.view.Window;
import android.view.WindowManager;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;

public class FrameStreamActivity extends AppCompatActivity implements CameraHelper.FrameProcessor {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1002;
    private static final float QUALITY_DISPLAY_THRESHOLD = 0.50f;
    private static final float EXPERIMENTAL_STACK_UPDATE_QUALITY_THRESHOLD = 0.75f;

    private static final int ITER_STATUS_CANDIDATE = 0;
    private static final int ITER_STATUS_SELECTED = 1;

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
    private Button btnToggleControls;
    private Button btnHideControls;
    private Button btnSettings;
    private androidx.appcompat.widget.SwitchCompat switchBypassQuality;
    private androidx.appcompat.widget.SwitchCompat switchExperimentalPrevReference;
    private androidx.appcompat.widget.SwitchCompat switchUseQuality;

    private View topBar;
    private View bottomDashboard;
    private View workspaceContainer;
    private View settingsPanel;
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
    private TextView lblReferenceIteration;
    private SeekBar seekBarReferenceIteration;
    private int selectedReferenceIteration = 0;
    private TextView lblHeatmapIteration;
    private SeekBar seekBarHeatmapIteration;
    private int selectedHeatmapIteration = 0;
    private TextView lblIterations;
    private SeekBar seekBarIterations;
    private volatile int iterationsNum = 4;
    private TextView lblQualityThreshold;
    private SeekBar seekBarQualityThreshold;
    private volatile boolean useQualityForTarget = true;
    private volatile float targetLostQualityThreshold = QUALITY_DISPLAY_THRESHOLD;

    private IterationResult[] cachedIterationResults = null;
    private Bitmap cachedFullFrameBmp = null;

    private CameraHelper cameraHelper;
    private boolean isLoopActive = false;
    private boolean controlsVisible = true;
    
    private float targetX = 0.0f;
    private float targetY = 0.0f;
    
    // Model Dimensions (Dynamic)
    private int searchW = 256;
    private int searchH = 256;
    private int heatmapW = 256;
    private int heatmapH = 256;
    private int refW = 64;
    private int refH = 64;
    private int refLayers = 16;
    private boolean refShapeIsNCHW = false;

    private int refInputIndex = 0;
    private int searchInputIndex = 1;
    private int heatmapOutputIndex = 0;
    private int qualityOutputIndex = 1;
    private String heatmapOutputDebug = "hm=?";
    private String qualityOutputDebug = "q=?";

    private java.nio.ByteBuffer refStackInputBuffer = null;
    private java.nio.ByteBuffer searchBuffer = null;
    private java.nio.ByteBuffer outputHeatmapBuffer = null;
    private java.nio.ByteBuffer outputQualityBuffer = null;

    private float[] initialReferenceStackLayers = null;
    private volatile ReferenceSource initialReferenceSource = null;
    private volatile float[] previousFrameReferenceStackLayers = null;
    private volatile ReferenceSource previousFrameReferenceSource = null;
    private volatile boolean experimentalPrevReferenceMode = false;

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        supportRequestWindowFeature(Window.FEATURE_NO_TITLE);
        enableFullscreenImmersive();
        setContentView(R.layout.activity_frame_stream);

        topBar = findViewById(R.id.top_bar);
        bottomDashboard = findViewById(R.id.bottom_dashboard);
        workspaceContainer = findViewById(R.id.workspace_container);
        viewFinder = findViewById(R.id.viewFinder);
        capturedImageView = findViewById(R.id.capturedImageView);
        tutorialHud = findViewById(R.id.tutorial_hud);
        lblStatus = findViewById(R.id.lbl_status);
        btnBack = findViewById(R.id.btn_back);
        btnToggleControls = findViewById(R.id.btn_toggle_controls);
        btnHideControls = findViewById(R.id.btn_hide_controls);
        btnSettings = findViewById(R.id.btn_settings);
        switchBypassQuality = findViewById(R.id.switch_bypass_quality);
        switchExperimentalPrevReference = findViewById(R.id.switch_experimental_prev_reference);
        switchUseQuality = findViewById(R.id.switch_use_quality);

        settingsPanel = findViewById(R.id.settings_panel);
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
        lblReferenceIteration = findViewById(R.id.lblReferenceIteration);
        seekBarReferenceIteration = findViewById(R.id.seekBarReferenceIteration);
        lblHeatmapIteration = findViewById(R.id.lblHeatmapIteration);
        seekBarHeatmapIteration = findViewById(R.id.seekBarHeatmapIteration);
        lblIterations = findViewById(R.id.lblIterations);
        seekBarIterations = findViewById(R.id.seekBarIterations);
        lblQualityThreshold = findViewById(R.id.lblQualityThreshold);
        seekBarQualityThreshold = findViewById(R.id.seekBarQualityThreshold);
        seekBarHistLayer.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                selectedHistLayer = progress;
                updateReferenceDebugLabel();
                if (cachedIterationResults != null) {
                    renderDiagnostics(cachedIterationResults, selectedHeatmapIteration, cachedFullFrameBmp);
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        seekBarReferenceIteration.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                selectedReferenceIteration = progress;
                updateReferenceDebugLabel();
                if (cachedIterationResults != null) {
                    renderDiagnostics(cachedIterationResults, selectedHeatmapIteration, cachedFullFrameBmp);
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        seekBarHeatmapIteration.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                selectedHeatmapIteration = progress;
                if (cachedIterationResults != null) {
                    renderDiagnostics(cachedIterationResults, selectedHeatmapIteration, cachedFullFrameBmp);
                } else {
                    lblHeatmapIteration.setText("Heatmap Iter: " + selectedHeatmapIteration);
                }
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        seekBarIterations.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                iterationsNum = progress + 1;
                lblIterations.setText("Iterations: " + iterationsNum);
                if (selectedHeatmapIteration >= iterationsNum) {
                    selectedHeatmapIteration = iterationsNum - 1;
                }
                if (selectedReferenceIteration >= iterationsNum) {
                    selectedReferenceIteration = iterationsNum - 1;
                }
                seekBarHeatmapIteration.setMax(Math.max(0, iterationsNum - 1));
                seekBarHeatmapIteration.setProgress(selectedHeatmapIteration);
                seekBarReferenceIteration.setMax(Math.max(0, iterationsNum - 1));
                seekBarReferenceIteration.setProgress(selectedReferenceIteration);
                updateReferenceDebugLabel();
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        seekBarIterations.setProgress(iterationsNum - 1);
        switchUseQuality.setChecked(useQualityForTarget);
        switchUseQuality.setOnCheckedChangeListener((buttonView, isChecked) -> {
            useQualityForTarget = isChecked;
            updateQualityThresholdLabel();
        });
        seekBarQualityThreshold.setProgress(Math.round(targetLostQualityThreshold * 100.0f));
        updateQualityThresholdLabel();
        seekBarQualityThreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                targetLostQualityThreshold = progress / 100.0f;
                updateQualityThresholdLabel();
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            tflite = new Interpreter(loadModelFile(), options);
            
            // Dynamically infer dimensions from model tensors
            int numInputs = tflite.getInputTensorCount();
            refInputIndex = -1;
            searchInputIndex = -1;
            for (int i = 0; i < numInputs; i++) {
                if (tensorNameContains(tflite.getInputTensor(i), "reference_stack") || tensorNameContains(tflite.getInputTensor(i), "ref")) {
                    refInputIndex = i;
                } else if (tensorNameContains(tflite.getInputTensor(i), "search_frame") || tensorNameContains(tflite.getInputTensor(i), "search")) {
                    searchInputIndex = i;
                }
            }
            // Fallback by shape check if names aren't resolved
            if (refInputIndex == -1 || searchInputIndex == -1 || refInputIndex == searchInputIndex) {
                if (numInputs >= 2) {
                    int[] shape0 = tflite.getInputTensor(0).shape();
                    int[] shape1 = tflite.getInputTensor(1).shape();
                    int dim0 = (shape0 != null && shape0.length > 1) ? shape0[shape0.length - 2] : 0;
                    int dim1 = (shape1 != null && shape1.length > 1) ? shape1[shape1.length - 2] : 0;
                    if (dim0 < dim1) {
                        refInputIndex = 0;
                        searchInputIndex = 1;
                    } else {
                        refInputIndex = 1;
                        searchInputIndex = 0;
                    }
                } else {
                    refInputIndex = 0;
                    searchInputIndex = 0;
                }
            }

            ParsedShape parsedRef = parseTensorShape(tflite.getInputTensor(refInputIndex), 64, 64, 16);
            refH = parsedRef.h;
            refW = parsedRef.w;
            refLayers = parsedRef.c;
            refShapeIsNCHW = parsedRef.isNCHW;
            
            ParsedShape parsedSearch = parseTensorShape(tflite.getInputTensor(searchInputIndex), 256, 256, 1);
            searchH = parsedSearch.h;
            searchW = parsedSearch.w;
            
            int numOutputs = tflite.getOutputTensorCount();
            heatmapOutputIndex = -1;
            qualityOutputIndex = -1;
            for (int i = 0; i < numOutputs; i++) {
                org.tensorflow.lite.Tensor outputTensor = tflite.getOutputTensor(i);
                if (tensorNameContains(outputTensor, "quality") || tensorNameContains(outputTensor, "predicted_quality")) {
                    qualityOutputIndex = i;
                } else if (tensorNameContains(outputTensor, "heatmap") || tensorNameContains(outputTensor, "predicted_heatmap")) {
                    heatmapOutputIndex = i;
                }
            }
            for (int i = 0; i < numOutputs && (heatmapOutputIndex == -1 || qualityOutputIndex == -1); i++) {
                int[] shape = tflite.getOutputTensor(i).shape();
                if (heatmapOutputIndex == -1 && shape != null && shape.length > 2 && shape[shape.length - 2] > 1) {
                    heatmapOutputIndex = i;
                } else if (qualityOutputIndex == -1 && i != heatmapOutputIndex) {
                    qualityOutputIndex = i;
                }
            }
            if (heatmapOutputIndex == -1) {
                heatmapOutputIndex = 0;
            }
            
            ParsedShape parsedHeatmap = parseTensorShape(tflite.getOutputTensor(heatmapOutputIndex), 256, 256, 1);
            heatmapH = parsedHeatmap.h;
            heatmapW = parsedHeatmap.w;
            heatmapOutputDebug = describeTensor("hm", heatmapOutputIndex);
            qualityOutputDebug = describeTensor("q", qualityOutputIndex);
            
            // Allocate buffers with dynamic sizes
            refStackInputBuffer = java.nio.ByteBuffer.allocateDirect(1 * refH * refW * refLayers * 4).order(java.nio.ByteOrder.nativeOrder());
            searchBuffer = java.nio.ByteBuffer.allocateDirect(searchH * searchW * 4).order(java.nio.ByteOrder.nativeOrder());
            outputHeatmapBuffer = java.nio.ByteBuffer.allocateDirect(heatmapH * heatmapW * 4).order(java.nio.ByteOrder.nativeOrder());
            outputQualityBuffer = java.nio.ByteBuffer.allocateDirect(4).order(java.nio.ByteOrder.nativeOrder());

            seekBarHistLayer.setMax(refLayers - 1);

            lblStatus.setText(String.format("Status: Engine loaded [%dx%d -> %dx%d]", searchW, searchH, heatmapW, heatmapH));
        } catch (Exception e) {
            e.printStackTrace();
            lblStatus.setText("Status: Failed to load TFLite model!");
        }

        btnBack.setOnClickListener(v -> finish());
        btnToggleControls.setOnClickListener(v -> setControlsVisible(true));
        btnHideControls.setOnClickListener(v -> setControlsVisible(false));
        btnSettings.setOnClickListener(v -> {
            settingsPanel.setVisibility(settingsPanel.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE);
            enableFullscreenImmersive();
        });
        switchExperimentalPrevReference.setOnCheckedChangeListener((buttonView, isChecked) -> {
            experimentalPrevReferenceMode = isChecked;
            previousFrameReferenceStackLayers = null;
            previousFrameReferenceSource = null;
        });
        setControlsVisible(false);
        
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

    private void setControlsVisible(boolean visible) {
        controlsVisible = visible;
        topBar.setVisibility(visible ? View.VISIBLE : View.GONE);
        bottomDashboard.setVisibility(visible ? View.VISIBLE : View.GONE);
        btnToggleControls.setVisibility(visible ? View.GONE : View.VISIBLE);

        androidx.constraintlayout.widget.ConstraintLayout.LayoutParams params =
                (androidx.constraintlayout.widget.ConstraintLayout.LayoutParams) workspaceContainer.getLayoutParams();
        if (visible) {
            params.topToTop = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
            params.topToBottom = R.id.top_bar;
            params.bottomToBottom = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
            params.bottomToTop = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
            params.startToStart = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.PARENT_ID;
            params.startToEnd = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
            params.endToStart = R.id.content_guideline;
            params.endToEnd = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
        } else {
            params.topToTop = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.PARENT_ID;
            params.topToBottom = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
            params.bottomToBottom = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.PARENT_ID;
            params.bottomToTop = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
            params.startToStart = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.PARENT_ID;
            params.startToEnd = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
            params.endToEnd = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.PARENT_ID;
            params.endToStart = androidx.constraintlayout.widget.ConstraintLayout.LayoutParams.UNSET;
        }
        workspaceContainer.setLayoutParams(params);
        enableFullscreenImmersive();
    }

    private void enableFullscreenImmersive() {
        if (getSupportActionBar() != null) {
            getSupportActionBar().hide();
        }
        getWindow().setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
        );
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        );
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

    private int canonicalReferenceIndex(int layer, int y, int x) {
        return layer * refH * refW + y * refW + x;
    }

    private int modelReferenceIndex(int layer, int y, int x) {
        if (refShapeIsNCHW) {
            return layer * refH * refW + y * refW + x;
        }
        return y * refW * refLayers + x * refLayers + layer;
    }

    private float getReferenceCropSizeForLayer(int layer, int frameHeight) {
        float smallestCropSize = (512.0f / 600.0f) * frameHeight;
        float largestCropSize = (512.0f / 600.0f) * frameHeight;
        if (refLayers <= 1) {
            return largestCropSize;
        }
        return largestCropSize - layer * ((largestCropSize - smallestCropSize) / (float)(refLayers - 1));
    }

    private float getReferenceCropSizeForIteration(int layer, int frameHeight, int iterationIndex) {
        float baseSize = getReferenceCropSizeForLayer(layer, frameHeight);
        if (iterationIndex <= 0) {
            return baseSize;
        }
        if (refLayers > 1 && layer == refLayers - 1) {
            return baseSize;
        }
        return Math.max(1.0f, baseSize * (float)Math.pow(0.5f, iterationIndex));
    }

    private float sampleYPlaneBilinear(byte[] yPlane, int width, int height, int stride, float srcX, float srcY) {
        int x0 = (int) Math.floor(srcX);
        int y0 = (int) Math.floor(srcY);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float dx = srcX - (float) x0;
        float dy = srcY - (float) y0;

        x0 = Math.max(0, Math.min(width - 1, x0));
        x1 = Math.max(0, Math.min(width - 1, x1));
        y0 = Math.max(0, Math.min(height - 1, y0));
        y1 = Math.max(0, Math.min(height - 1, y1));

        float p00 = yPlane[y0 * stride + x0] & 0xFF;
        float p10 = yPlane[y0 * stride + x1] & 0xFF;
        float p01 = yPlane[y1 * stride + x0] & 0xFF;
        float p11 = yPlane[y1 * stride + x1] & 0xFF;

        return ((1.0f - dx) * (1.0f - dy) * p00)
                + (dx * (1.0f - dy) * p10)
                + ((1.0f - dx) * dy * p01)
                + (dx * dy * p11);
    }

    private float[] buildReferenceStackLayers(byte[] yPlane, int width, int height, int stride, float cx, float cy) {
        return buildReferenceStackLayers(yPlane, width, height, stride, cx, cy, 0);
    }

    private float[] buildReferenceStackLayers(ReferenceSource source, int iterationIndex) {
        if (source == null) {
            return null;
        }
        return buildReferenceStackLayers(
                source.yPlane,
                source.width,
                source.height,
                source.stride,
                source.cx,
                source.cy,
                iterationIndex
        );
    }

    private float[] buildReferenceStackLayers(byte[] yPlane, int width, int height, int stride, float cx, float cy, int iterationIndex) {
        float[] stack = new float[refLayers * refH * refW];
        for (int layer = 0; layer < refLayers; layer++) {
            float size = getReferenceCropSizeForIteration(layer, height, iterationIndex);
            float half = size / 2.0f;

            for (int y = 0; y < refH; y++) {
                for (int x = 0; x < refW; x++) {
                    float srcX = cx - half + (x / (float)(refW - 1)) * size;
                    float srcY = cy - half + (y / (float)(refH - 1)) * size;

                    float pixel = sampleYPlaneBilinear(yPlane, width, height, stride, srcX, srcY);
                    stack[canonicalReferenceIndex(layer, y, x)] = pixel / 255.0f;
                }
            }
        }
        return stack;
    }

    private void preserveInitialSmallestReferenceLayer(float[] stackLayers) {
        if (stackLayers == null || initialReferenceStackLayers == null || refLayers <= 1) {
            return;
        }

        int smallestLayer = refLayers - 1;
        for (int y = 0; y < refH; y++) {
            for (int x = 0; x < refW; x++) {
                int index = canonicalReferenceIndex(smallestLayer, y, x);
                stackLayers[index] = initialReferenceStackLayers[index];
            }
        }
    }

    private ReferenceSource selectReferenceSourceForCurrentFrame() {
        if (experimentalPrevReferenceMode && previousFrameReferenceSource != null) {
            return previousFrameReferenceSource;
        }
        return initialReferenceSource;
    }

    private float[] calculateDiscreteArgmaxCoords(float[] heatmap, int hmW, int hmH) {
        HeatmapStats stats = calculateHeatmapStats(heatmap, hmW, hmH);

        return new float[] { stats.maxX / (float) hmW, stats.maxY / (float) hmH };
    }

    private HeatmapStats calculateHeatmapStats(float[] heatmap, int hmW, int hmH) {
        float minVal = Float.MAX_VALUE;
        float maxVal = -Float.MAX_VALUE;
        int maxX = hmW / 2;
        int maxY = hmH / 2;

        for (int y = 0; y < hmH; y++) {
            for (int x = 0; x < hmW; x++) {
                float val = heatmap[y * hmW + x];
                if (val < minVal) {
                    minVal = val;
                }
                if (val > maxVal) {
                    maxVal = val;
                    maxX = x;
                    maxY = y;
                }
            }
        }

        return new HeatmapStats(minVal, maxVal, maxX, maxY);
    }

    private int selectBestQualityIteration(IterationResult[] iterationResults) {
        if (iterationResults == null || iterationResults.length == 0) {
            return -1;
        }

        for (IterationResult result : iterationResults) {
            result.status = ITER_STATUS_CANDIDATE;
        }

        int selectedIndex = 0;
        float bestQuality = iterationResults[0].qualityScore;
        for (int i = 1; i < iterationResults.length; i++) {
            if (iterationResults[i].qualityScore > bestQuality) {
                bestQuality = iterationResults[i].qualityScore;
                selectedIndex = i;
            }
        }

        iterationResults[selectedIndex].status = ITER_STATUS_SELECTED;
        return selectedIndex;
    }

    private void writeReferenceStackToInputBuffer(float[] stackLayers) {
        if (stackLayers == null || refStackInputBuffer == null) return;

        refStackInputBuffer.rewind();
        java.nio.FloatBuffer refFloatBuffer = refStackInputBuffer.asFloatBuffer();
        for (int layer = 0; layer < refLayers; layer++) {
            for (int y = 0; y < refH; y++) {
                for (int x = 0; x < refW; x++) {
                    refFloatBuffer.put(
                            modelReferenceIndex(layer, y, x),
                            stackLayers[canonicalReferenceIndex(layer, y, x)]
                    );
                }
            }
        }
        refStackInputBuffer.rewind();
    }

    private void gatherReferenceFrame(byte[] yPlane, int width, int height) {
        float[] normCoords = MainActivity.mapAlignedScreenCoordsToFrame(targetX, targetY, viewFinder.getWidth(), viewFinder.getHeight(), width, height);
        if (normCoords == null) {
            currentUiState = STATE_IDLE;
            return;
        }
        
        float cx = normCoords[0] * width;
        float cy = normCoords[1] * height;

        int stride = width; // rotated frame stride is width

        initialReferenceSource = new ReferenceSource(yPlane, width, height, stride, cx, cy);
        initialReferenceStackLayers = buildReferenceStackLayers(initialReferenceSource, 0);
        previousFrameReferenceStackLayers = null;
        previousFrameReferenceSource = null;
        writeReferenceStackToInputBuffer(initialReferenceStackLayers);
        
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
        
        float[] currBuffer = new float[searchW * searchH];
        long preStart = SystemClock.elapsedRealtime();
        
        // Search input is always the centered square crop of the current frame.
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        
        // JNI extracts square crop around static frame center (cx, cy)
        MainActivity.downsampleSearchCrop(yPlane, width, height, stride, cx, cy, cropSize, searchW, searchH, currBuffer);
        
        long preDuration = SystemClock.elapsedRealtime() - preStart;

        ReferenceSource referenceSource = selectReferenceSourceForCurrentFrame();
        if (referenceSource == null) return;

        int activeIterations = Math.max(1, iterationsNum);
        IterationResult[] iterationResults = new IterationResult[activeIterations];
        long infDuration = 0L;
        long postDuration = 0L;

        boolean hasQualityOutput = (qualityOutputIndex >= 0 && outputQualityBuffer != null);
        Object[] inputs = new Object[2];
        inputs[refInputIndex] = refStackInputBuffer;
        inputs[searchInputIndex] = searchBuffer;

        for (int iterIdx = 0; iterIdx < activeIterations; iterIdx++) {
            float[] iterReferenceStack = buildReferenceStackLayers(referenceSource, iterIdx);
            preserveInitialSmallestReferenceLayer(iterReferenceStack);
            writeReferenceStackToInputBuffer(iterReferenceStack);

            searchBuffer.rewind();
            searchBuffer.asFloatBuffer().put(currBuffer);
            searchBuffer.rewind();

            outputHeatmapBuffer.rewind();
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(heatmapOutputIndex, outputHeatmapBuffer);
            if (hasQualityOutput) {
                outputQualityBuffer.rewind();
                outputs.put(qualityOutputIndex, outputQualityBuffer);
            }

            long infStart = SystemClock.elapsedRealtime();
            tflite.runForMultipleInputsOutputs(inputs, outputs);
            infDuration += SystemClock.elapsedRealtime() - infStart;

            float[] outputHeatmap = new float[heatmapW * heatmapH];
            outputHeatmapBuffer.rewind();
            outputHeatmapBuffer.asFloatBuffer().get(outputHeatmap);

            long postStart = SystemClock.elapsedRealtime();
            HeatmapStats heatmapStats = calculateHeatmapStats(outputHeatmap, heatmapW, heatmapH);
            float iterPxNorm = heatmapStats.maxX / (float) heatmapW;
            float iterPyNorm = heatmapStats.maxY / (float) heatmapH;
            postDuration += SystemClock.elapsedRealtime() - postStart;

            float qualityScore = hasQualityOutput ? outputQualityBuffer.getFloat(0) : heatmapStats.maxVal;
            iterationResults[iterIdx] = new IterationResult(
                    Arrays.copyOf(currBuffer, currBuffer.length),
                    Arrays.copyOf(iterReferenceStack, iterReferenceStack.length),
                    outputHeatmap,
                    heatmapStats,
                    iterPxNorm,
                    iterPyNorm,
                    iterPxNorm,
                    iterPyNorm,
                    qualityScore
            );

        }

        final int selectedResultIndex = Math.max(0, selectBestQualityIteration(iterationResults));
        final IterationResult finalIterationResult = iterationResults[selectedResultIndex];
        final HeatmapStats heatmapStats = finalIterationResult.heatmapStats;
        final float finalQualityScore = finalIterationResult.qualityScore;
        float px = finalIterationResult.mappedPxNorm;
        float py = finalIterationResult.mappedPyNorm;
        
        // Coordinate Re-projection: Map crop-relative coordinates back to camera absolute coordinates using static center
        float halfSize = cropSize / 2.0f;
        float srcX_start = cx - halfSize;
        float srcY_start = cy - halfSize;
        float x_global = srcX_start + px * cropSize;
        float y_global = srcY_start + py * cropSize;

        // Clamp target position to camera frame boundaries to keep tracking running continuously
        x_global = Math.max(0.0f, Math.min((float)width - 1.0f, x_global));
        y_global = Math.max(0.0f, Math.min((float)height - 1.0f, y_global));
        final float finalTrackedX = x_global;
        final float finalTrackedY = y_global;

        final boolean finalExperimentalMode = experimentalPrevReferenceMode;
        final boolean finalStackUpdated = finalExperimentalMode
                && finalQualityScore >= EXPERIMENTAL_STACK_UPDATE_QUALITY_THRESHOLD;
        if (finalStackUpdated) {
            ReferenceSource updatedReferenceSource = new ReferenceSource(yPlane, width, height, stride, finalTrackedX, finalTrackedY);
            float[] updatedStackLayers = buildReferenceStackLayers(updatedReferenceSource, 0);
            preserveInitialSmallestReferenceLayer(updatedStackLayers);
            previousFrameReferenceSource = updatedReferenceSource;
            previousFrameReferenceStackLayers = updatedStackLayers;
        }

        long totalDuration = SystemClock.elapsedRealtime() - startTime;
        final long finalInfDuration = infDuration;
        final long finalPostDuration = postDuration;

        // Convert absolute camera coordinates to normalized [0, 1] relative to camera frame
        float gx = finalTrackedX / (float) width;
        float gy = finalTrackedY / (float) height;

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
            float scaledX = finalTrackedX / (float) downsampleFactor;
            float scaledY = finalTrackedY / (float) downsampleFactor;
            canvas.drawCircle(scaledX, scaledY, 3.0f, paint);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(1.0f);
            paint.setColor(Color.CYAN);
            float scaledCropSize = cropSize / (float) downsampleFactor;
            float scaledLeft = (smallW - scaledCropSize) / 2.0f;
            float scaledTop = (smallH - scaledCropSize) / 2.0f;
            canvas.drawRect(scaledLeft, scaledTop, scaledLeft + scaledCropSize, scaledTop + scaledCropSize, paint);
        } catch (Exception e) {
            e.printStackTrace();
        }
        final Bitmap fullFrameBmp = tmpBmp;

        // Save to cache for SeekBar scrolling redraws
        cachedIterationResults = iterationResults;
        cachedFullFrameBmp = fullFrameBmp;

        // Post UI rendering tasks back to the UI thread
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                if (currentUiState != STATE_TRACKING) return;

                int maxHeatmapIteration = Math.max(0, iterationResults.length - 1);
                seekBarHeatmapIteration.setMax(maxHeatmapIteration);
                seekBarReferenceIteration.setMax(maxHeatmapIteration);
                if (selectedHeatmapIteration > maxHeatmapIteration) {
                    selectedHeatmapIteration = maxHeatmapIteration;
                    seekBarHeatmapIteration.setProgress(selectedHeatmapIteration);
                }
                if (selectedReferenceIteration > maxHeatmapIteration) {
                    selectedReferenceIteration = maxHeatmapIteration;
                    seekBarReferenceIteration.setProgress(selectedReferenceIteration);
                }
                updateReferenceDebugLabel();
                
                float targetColorThreshold = targetLostQualityThreshold;
                boolean targetLostByQuality = useQualityForTarget && finalQualityScore < targetColorThreshold;
                int circleColor = targetLostByQuality ? Color.RED : Color.GREEN;
                lblStatus.setText(targetLostByQuality
                        ? String.format("Status: Weak/Lost Target | Quality: %.2f < %.2f", finalQualityScore, targetColorThreshold)
                        : "Status: Active tracking");
                drawTrackingIndicator(finalTrackedX, finalTrackedY, width, height, circleColor);
                renderDiagnostics(iterationResults, selectedHeatmapIteration, fullFrameBmp);

                txtLatency.setText(String.format("Latency: Pre:%dms | TFLite:%dms | CoM:%dms (Total:%dms)", 
                        preDuration, finalInfDuration, finalPostDuration, totalDuration));
                txtPredictedCoords.setText(String.format(
                        "Target Pos: (%.3f, %.3f) [Quality: %.2f | Selected: %d | Stack update: %s]",
                        screenX_norm,
                        screenY_norm,
                        finalQualityScore,
                        selectedResultIndex,
                        finalStackUpdated ? "yes" : "no"
                ));
                txtBufferStatus.setText(String.format(
                        "Heatmap raw: min=%.3f max=%.3f argmax=(%d,%d) p=(%.3f,%.3f) search_center=(%.0f,%.0f) global=(%.0f,%.0f) | %s | %s",
                        heatmapStats.minVal,
                        heatmapStats.maxVal,
                        heatmapStats.maxX,
                        heatmapStats.maxY,
                        px,
                        py,
                        cx,
                        cy,
                        finalTrackedX,
                        finalTrackedY,
                        heatmapOutputDebug,
                        qualityOutputDebug
                ));
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
        drawSearchFrameBoundary(canvas, imgW, imgH);

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

    private void drawSearchFrameBoundary(Canvas canvas, int imgW, int imgH) {
        float activeSize = Math.min(imgW, imgH);
        float left = (imgW - activeSize) / 2.0f;
        float top = (imgH - activeSize) / 2.0f;
        float right = left + activeSize;
        float bottom = top + activeSize;

        Paint deadPaint = new Paint();
        deadPaint.setStyle(Paint.Style.FILL);
        deadPaint.setColor(Color.argb(70, 255, 80, 0));
        if (top > 0.0f) {
            canvas.drawRect(0.0f, 0.0f, imgW, top, deadPaint);
        }
        if (bottom < imgH) {
            canvas.drawRect(0.0f, bottom, imgW, imgH, deadPaint);
        }
        if (left > 0.0f) {
            canvas.drawRect(0.0f, top, left, bottom, deadPaint);
        }
        if (right < imgW) {
            canvas.drawRect(right, top, imgW, bottom, deadPaint);
        }

        Paint borderPaint = new Paint();
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setAntiAlias(true);
        borderPaint.setStrokeWidth(Math.max(2.0f, imgW / 360.0f));
        borderPaint.setColor(Color.CYAN);
        canvas.drawRect(left, top, right, bottom, borderPaint);
    }

    private void renderDiagnostics(IterationResult[] iterationResults, int selectedIteration, Bitmap fullFrameBmp) {
        if (iterationResults == null || iterationResults.length == 0) {
            return;
        }

        int safeIteration = Math.max(0, Math.min(selectedIteration, iterationResults.length - 1));
        IterationResult selectedResult = iterationResults[safeIteration];
        int safeReferenceIteration = Math.max(0, Math.min(selectedReferenceIteration, iterationResults.length - 1));
        IterationResult referenceResult = iterationResults[safeReferenceIteration];
        float[] heatmap = selectedResult.heatmap;
        float[] curr = selectedResult.search;
        float px = selectedResult.iterPxNorm;
        float py = selectedResult.iterPyNorm;
        lblHeatmapIteration.setText(String.format(
                "Heatmap Iter: %d/%d | q=%.2f | %s",
                safeIteration,
                iterationResults.length - 1,
                selectedResult.qualityScore,
                iterationStatusLabel(selectedResult.status)
        ));

        // 1. Render raw output heatmap with display-only min/max normalization.
        Bitmap hmBitmap = Bitmap.createBitmap(heatmapW, heatmapH, Bitmap.Config.ARGB_8888);
        int[] hmColors = new int[heatmapW * heatmapH];
        HeatmapStats stats = selectedResult.heatmapStats;
        float denom = stats.maxVal - stats.minVal;
        if (Math.abs(denom) < 1e-6f) {
            denom = 1.0f;
        }
        for (int i = 0; i < heatmapW * heatmapH; i++) {
            float val = (heatmap[i] - stats.minVal) / denom;
            val = Math.max(0.0f, Math.min(val, 1.0f));
            int r = (int)(val * 255.0f);
            int b = (int)((1.0f - val) * 255.0f);
            int g = (int)(val * 100.0f);
            hmColors[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        hmBitmap.setPixels(hmColors, 0, heatmapW, 0, 0, heatmapW, heatmapH);
        Canvas hmCanvas = new Canvas(hmBitmap);
        Paint hmPaint = new Paint();
        hmPaint.setAntiAlias(true);
        hmPaint.setStyle(Paint.Style.STROKE);
        hmPaint.setStrokeWidth(Math.max(1.0f, heatmapW / 64.0f));
        hmPaint.setColor(Color.GREEN);
        hmCanvas.drawCircle(stats.maxX, stats.maxY, Math.max(2.0f, heatmapW / 32.0f), hmPaint);
        hmCanvas.drawLine(stats.maxX - 3.0f, stats.maxY, stats.maxX + 3.0f, stats.maxY, hmPaint);
        hmCanvas.drawLine(stats.maxX, stats.maxY - 3.0f, stats.maxX, stats.maxY + 3.0f, hmPaint);
        heatmapImageView.setImageBitmap(hmBitmap);

        // 2. Render cropCurrView (the current active search crop)
        Bitmap currBitmap = Bitmap.createBitmap(searchW, searchH, Bitmap.Config.ARGB_8888);
        int[] currColors = new int[searchW * searchH];
        for (int i = 0; i < searchW * searchH; i++) {
            int val = (int)(curr[i] * 255.0f);
            currColors[i] = 0xFF000000 | (val << 16) | (val << 8) | val;
        }
        currBitmap.setPixels(currColors, 0, searchW, 0, 0, searchW, searchH);
        
        // Draw a small red indicator showing the predicted target position inside search crop
        Canvas canvasCurr = new Canvas(currBitmap);
        Paint paintCurr = new Paint();
        paintCurr.setColor(Color.RED);
        paintCurr.setStyle(Paint.Style.FILL);
        paintCurr.setAntiAlias(true);
        canvasCurr.drawCircle(px * searchW, py * searchH, 6.0f, paintCurr);
        
        cropCurrView.setImageBitmap(currBitmap);

        // 3. Render cropHistView (the locked target reference template layer)
        Bitmap histBitmap = Bitmap.createBitmap(refW, refH, Bitmap.Config.ARGB_8888);
        int[] histColors = new int[refW * refH];
        float[] refStackForDisplay = referenceResult.referenceStack;
        for (int y = 0; y < refH; y++) {
            for (int x = 0; x < refW; x++) {
                int index = canonicalReferenceIndex(selectedHistLayer, y, x);
                int val = (int)(refStackForDisplay[index] * 255.0f);
                histColors[y * refW + x] = 0xFF000000 | (val << 16) | (val << 8) | val;
            }
        }
        histBitmap.setPixels(histColors, 0, refW, 0, 0, refW, refH);
        
        // Draw a small red indicator at the center of the reference crop template (the target origin)
        Canvas canvasHist = new Canvas(histBitmap);
        Paint paintHist = new Paint();
        paintHist.setColor(Color.RED);
        paintHist.setStyle(Paint.Style.FILL);
        paintHist.setAntiAlias(true);
        canvasHist.drawCircle(refW / 2.0f, refH / 2.0f, 3.0f, paintHist);
        
        cropHistView.setImageBitmap(histBitmap);

        // 4. Render cropFullView (the downsampled full processed frame)
        if (fullFrameBmp != null) {
            cropFullView.setImageBitmap(fullFrameBmp);
        }
    }

    private String iterationStatusLabel(int status) {
        switch (status) {
            case ITER_STATUS_SELECTED:
                return "SELECTED";
            default:
                return "CANDIDATE";
        }
    }

    private void updateQualityThresholdLabel() {
        if (lblQualityThreshold == null) {
            return;
        }
        String mode = useQualityForTarget ? "on" : "off";
        lblQualityThreshold.setText(String.format("Quality lost threshold: %.2f (%s)", targetLostQualityThreshold, mode));
    }

    private void updateReferenceDebugLabel() {
        if (lblHistStack == null) {
            return;
        }
        lblHistStack.setText(String.format("Hist Stack (I: %d, L: %d)", selectedReferenceIteration, selectedHistLayer));
        if (lblReferenceIteration != null) {
            lblReferenceIteration.setText("Ref Iter: " + selectedReferenceIteration);
        }
    }

    private void resetTrackerToIdle() {
        currentUiState = STATE_IDLE;
        resultsPanel.setVisibility(View.GONE);
        initialReferenceStackLayers = null;
        initialReferenceSource = null;
        previousFrameReferenceStackLayers = null;
        previousFrameReferenceSource = null;
        
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
        enableFullscreenImmersive();
        startWorkerThread();
        if (cameraHelper != null && cameraHelper.hasCameraPermission() && !isLoopActive) {
            isLoopActive = true;
        }
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            enableFullscreenImmersive();
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

    private static class ReferenceSource {
        final byte[] yPlane;
        final int width;
        final int height;
        final int stride;
        final float cx;
        final float cy;

        ReferenceSource(byte[] yPlane, int width, int height, int stride, float cx, float cy) {
            this.yPlane = Arrays.copyOf(yPlane, yPlane.length);
            this.width = width;
            this.height = height;
            this.stride = stride;
            this.cx = cx;
            this.cy = cy;
        }
    }

    private static class ParsedShape {
        int h;
        int w;
        int c;
        boolean isNCHW;

        ParsedShape(int h, int w, int c, boolean isNCHW) {
            this.h = h;
            this.w = w;
            this.c = c;
            this.isNCHW = isNCHW;
        }
    }

    private static class HeatmapStats {
        final float minVal;
        final float maxVal;
        final int maxX;
        final int maxY;

        HeatmapStats(float minVal, float maxVal, int maxX, int maxY) {
            this.minVal = minVal;
            this.maxVal = maxVal;
            this.maxX = maxX;
            this.maxY = maxY;
        }
    }

    private static class IterationResult {
        final float[] search;
        final float[] referenceStack;
        final float[] heatmap;
        final HeatmapStats heatmapStats;
        final float iterPxNorm;
        final float iterPyNorm;
        final float mappedPxNorm;
        final float mappedPyNorm;
        final float qualityScore;
        int status = ITER_STATUS_CANDIDATE;

        IterationResult(
                float[] search,
                float[] referenceStack,
                float[] heatmap,
                HeatmapStats heatmapStats,
                float iterPxNorm,
                float iterPyNorm,
                float mappedPxNorm,
                float mappedPyNorm,
                float qualityScore) {
            this.search = search;
            this.referenceStack = referenceStack;
            this.heatmap = heatmap;
            this.heatmapStats = heatmapStats;
            this.iterPxNorm = iterPxNorm;
            this.iterPyNorm = iterPyNorm;
            this.mappedPxNorm = mappedPxNorm;
            this.mappedPyNorm = mappedPyNorm;
            this.qualityScore = qualityScore;
        }
    }

    private ParsedShape parseTensorShape(org.tensorflow.lite.Tensor tensor, int defaultH, int defaultW, int defaultC) {
        if (tensor == null) {
            return new ParsedShape(defaultH, defaultW, defaultC, false);
        }
        int[] shape = tensor.shape();
        if (shape == null || shape.length < 3) {
            return new ParsedShape(defaultH, defaultW, defaultC, false);
        }

        int h, w, c;
        boolean isNCHW = false;

        if (shape.length == 4) {
            if (shape[1] == shape[2]) {
                // NHWC: [batch, H, W, C]
                h = shape[1];
                w = shape[2];
                c = shape[3];
                isNCHW = false;
            } else if (shape[2] == shape[3]) {
                // NCHW: [batch, C, H, W]
                c = shape[1];
                h = shape[2];
                w = shape[3];
                isNCHW = true;
            } else {
                // Fallback to NHWC
                h = shape[1];
                w = shape[2];
                c = shape[3];
                isNCHW = false;
            }
        } else { // length == 3
            if (shape[0] == shape[1]) {
                // HWC: [H, W, C]
                h = shape[0];
                w = shape[1];
                c = shape[2];
                isNCHW = false;
            } else if (shape[1] == shape[2]) {
                // CHW: [C, H, W]
                c = shape[0];
                h = shape[1];
                w = shape[2];
                isNCHW = true;
            } else {
                h = shape[0];
                w = shape[1];
                c = shape[2];
                isNCHW = false;
            }
        }
        return new ParsedShape(h, w, c, isNCHW);
    }

    private boolean tensorNameContains(org.tensorflow.lite.Tensor tensor, String query) {
        if (tensor == null) return false;
        String name = tensor.name();
        if (name == null) return false;
        return name.toLowerCase().contains(query.toLowerCase());
    }

    private String describeTensor(String label, int index) {
        if (tflite == null || index < 0 || index >= tflite.getOutputTensorCount()) {
            return label + "=none";
        }
        org.tensorflow.lite.Tensor tensor = tflite.getOutputTensor(index);
        String name = tensor.name() == null ? "unnamed" : tensor.name();
        return String.format("%s=%d:%s%s", label, index, name, Arrays.toString(tensor.shape()));
    }
}
