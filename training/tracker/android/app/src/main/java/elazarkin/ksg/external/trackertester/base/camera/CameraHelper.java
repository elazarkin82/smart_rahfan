package elazarkin.ksg.external.trackertester.base.camera;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;

import com.google.common.util.concurrent.ListenableFuture;

public class CameraHelper {

    private final LifecycleOwner lifecycleOwner;
    private final PreviewView previewView;
    private ProcessCameraProvider cameraProvider;

    public interface OnCameraReadyCallback {
        void onCameraReady();
        void onCameraError(Exception e);
    }

    public CameraHelper(LifecycleOwner lifecycleOwner, PreviewView previewView) {
        this.lifecycleOwner = lifecycleOwner;
        this.previewView = previewView;
    }

    /**
     * Checks if camera permission is currently granted.
     */
    public boolean hasCameraPermission() {
        Context context = previewView.getContext();
        return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) 
                == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Requests camera permission from the specified Activity.
     */
    public void requestCameraPermission(Activity activity, int requestCode) {
        ActivityCompat.requestPermissions(
                activity, 
                new String[]{Manifest.permission.CAMERA}, 
                requestCode
        );
    }

    /**
     * Starts the CameraX preview stream and binds it to the specified LifecycleOwner.
     */
    public void startCamera(final OnCameraReadyCallback callback) {
        Context context = previewView.getContext();
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = 
                ProcessCameraProvider.getInstance(context);

        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
                if (callback != null) {
                    callback.onCameraReady();
                }
            } catch (Exception e) {
                e.printStackTrace();
                if (callback != null) {
                    callback.onCameraError(e);
                }
            }
        }, ContextCompat.getMainExecutor(context));
    }

    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
        
        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview);
    }

    /**
     * Grabs the current preview frame as a Bitmap.
     */
    public Bitmap captureFrame() {
        return previewView.getBitmap();
    }

    /**
     * Explicitly unbinds the camera provider resources.
     */
    public void shutdown() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
    }
}
