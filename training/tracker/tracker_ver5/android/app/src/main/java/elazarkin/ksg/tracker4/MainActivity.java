package elazarkin.ksg.tracker4;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;


import elazarkin.ksg.tracker4.activities.FrameStreamActivity;
import elazarkin.ksg.tracker4.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'tracker4' library on application startup.
    static {
        System.loadLibrary("tracker4");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Set click listener for the dashboard card
        binding.cardAutoCorrelation.setVisibility(View.GONE);

        binding.cardFrameStream.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, FrameStreamActivity.class);
                startActivity(intent);
            }
        });
    }

    /**
     * Native JNI pre-processing function that extracts a square crop from the Y-plane,
     * applies bilinear interpolation and replication padding, and downsamples it to a target Float32 array.
     */
    public static native void downsampleSearchCrop(
            byte[] yPlane,
            int srcW,
            int srcH,
            int rowStride,
            float cx,
            float cy,
            float cropSize,
            int outW,
            int outH,
            float[] outBuffer
    );

    /**
     * Native JNI pre-processing function that rotates the Y-plane byte array
     * according to camera rotation.
     */
    public static native void rotateYPlane(
            byte[] src,
            byte[] dest,
            int width,
            int height,
            int stride,
            int rotationDegrees
    );

    /**
     * Native JNI post-processing function that calculates the noise-immune Local Refined Argmax Centroid on the predicted heatmap.
     */
    public static native float[] calculateLocalRefinedArgmaxCentroid(
            float[] heatmap,
            int hmW,
            int hmH
    );

    /**
     * Maps screen touch coordinates (viewX, viewY) relative to the PreviewView
     * into normalized [0, 1] coordinates inside the captured frame Bitmap using FIT_CENTER,
     * taking camera sensor rotation degrees into account.
     */
    public static float[] mapScreenCoordsToFrame(float viewX, float viewY, int viewWidth, int viewHeight, int imgWidth, int imgHeight, int rotationDegrees) {
        if (viewWidth <= 0 || viewHeight <= 0 || imgWidth <= 0 || imgHeight <= 0) return null;
        
        // 1. Determine effective image dimensions on the portrait screen based on camera rotation
        int effectiveImgW = (rotationDegrees == 90 || rotationDegrees == 270) ? imgHeight : imgWidth;
        int effectiveImgH = (rotationDegrees == 90 || rotationDegrees == 270) ? imgWidth : imgHeight;
        
        float viewRatio = (float) viewWidth / viewHeight;
        float imgRatio = (float) effectiveImgW / effectiveImgH;
        
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        
        if (imgRatio > viewRatio) { // Fit Width, height is letterboxed
            float actualHeight = viewWidth / imgRatio;
            offsetY = (viewHeight - actualHeight) / 2.0f;
            scaleX = 1.0f / viewWidth;
            scaleY = 1.0f / actualHeight;
        } else { // Fit Height, width is letterboxed
            float actualWidth = viewHeight * imgRatio;
            offsetX = (viewWidth - actualWidth) / 2.0f;
            scaleX = 1.0f / actualWidth;
            scaleY = 1.0f / viewHeight;
        }
        
        // 2. Map screen coordinate to normalized displayed image space [0.0, 1.0]
        float normX = (viewX - offsetX) * scaleX;
        float normY = (viewY - offsetY) * scaleY;
        
        if (normX < 0.0f || normX > 1.0f || normY < 0.0f || normY > 1.0f) {
            return null;
        }
        
        // 3. Rotate normalized displayed coordinates back to the sensor coordinate space
        float sensorX_norm;
        float sensorY_norm;
        if (rotationDegrees == 90) {
            sensorX_norm = normY;
            sensorY_norm = 1.0f - normX;
        } else if (rotationDegrees == 180) {
            sensorX_norm = 1.0f - normX;
            sensorY_norm = 1.0f - normY;
        } else if (rotationDegrees == 270) {
            sensorX_norm = 1.0f - normY;
            sensorY_norm = normX;
        } else {
            sensorX_norm = normX;
            sensorY_norm = normY;
        }
        
        return new float[]{ sensorX_norm, sensorY_norm };
    }

    /**
     * Maps screen touch coordinates (viewX, viewY) relative to the PreviewView
     * into normalized [0, 1] coordinates inside the already rotated/aligned frame.
     */
    public static float[] mapAlignedScreenCoordsToFrame(float viewX, float viewY, int viewWidth, int viewHeight, int imgWidth, int imgHeight) {
        if (viewWidth <= 0 || viewHeight <= 0 || imgWidth <= 0 || imgHeight <= 0) return null;
        
        float viewRatio = (float) viewWidth / viewHeight;
        float imgRatio = (float) imgWidth / imgHeight;
        
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        
        if (imgRatio > viewRatio) { // Fit Width, height is letterboxed
            float actualHeight = viewWidth / imgRatio;
            offsetY = (viewHeight - actualHeight) / 2.0f;
            scaleX = 1.0f / viewWidth;
            scaleY = 1.0f / actualHeight;
        } else { // Fit Height, width is letterboxed
            float actualWidth = viewHeight * imgRatio;
            offsetX = (viewWidth - actualWidth) / 2.0f;
            scaleX = 1.0f / actualWidth;
            scaleY = 1.0f / viewHeight;
        }
        
        float normX = (viewX - offsetX) * scaleX;
        float normY = (viewY - offsetY) * scaleY;
        
        if (normX < 0.0f || normX > 1.0f || normY < 0.0f || normY > 1.0f) {
            return null;
        }
        return new float[]{ normX, normY };
    }

    public native String stringFromJNI();
}